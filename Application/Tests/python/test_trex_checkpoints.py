"""Recognition checkpoint compatibility using real Torch serializers on CPU."""

import copy
import importlib
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest import mock

import numpy as np
import torch


RUNTIME_DIR = Path(__file__).resolve().parents[2] / "src" / "tracker" / "python"
sys.path.insert(0, str(RUNTIME_DIR))


class FakeVIWeights:
    def __init__(self, **fields):
        self.fields = fields
        self.path = types.SimpleNamespace(str=lambda: fields["path"])

    def to_json(self):
        return json.dumps(self.fields)

    def to_string(self):
        return self.to_json()


class CheckpointTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        trex = types.ModuleType("TRex")
        trex.log = lambda *args: None
        trex.warn = lambda *args: None
        trex.setting = lambda name: "checkpoint-test"
        trex.choose_device = lambda: "cpu"
        trex.VIWeights = FakeVIWeights
        trex.DetectResolution = list
        modules = mock.patch.dict(sys.modules, {"TRex": trex})
        modules.start()
        cls.addClassCleanup(modules.stop)
        cls.utils = importlib.import_module("trex_utils")
        cls.recognition = importlib.import_module("visual_recognition_torch")

    def setUp(self):
        temporary = tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent)
        self.addCleanup(temporary.cleanup)
        self.base = str(Path(temporary.name) / "weights")
        settings = mock.patch.multiple(
            self.recognition, create=True,
            image_width=24, image_height=16, image_channels=1, classes=[0, 1, 2],
            network_version="v118_3", output_path=self.base, loaded_weights=None,
            loaded_checkpoint=None, model=None, p_softmax=None,
        )
        settings.start()
        self.addCleanup(settings.stop)
        self.model = self.recognition.get_default_network()
        self.metadata = {"input_shape": [24, 16, 1], "num_classes": 3,
                         "model_type": "v118_3", "uniqueness": 0.75, "epoch": 4}

    def assert_weights_equal(self, actual, expected):
        self.assertEqual(set(actual), set(expected))
        for key in expected:
            torch.testing.assert_close(actual[key], expected[key])

    def test_legacy_weights_do_not_enter_jit_loader(self):
        state = self.model.state_dict()
        for raw, old_serialization in ((False, False), (True, False), (True, True)):
            with self.subTest(raw=raw, old_serialization=old_serialization):
                path = self.base + ".pth"
                checkpoint = state if raw else {
                    "model": None, "state_dict": state, "metadata": self.metadata}
                torch.save(checkpoint, path,
                           _use_new_zipfile_serialization=not old_serialization)
                with mock.patch.object(torch.jit, "load", side_effect=AssertionError("Unexpected JIT load")):
                    loaded, _ = self.recognition.load_model_from_file(path, "cpu")
                self.assert_weights_equal(loaded.state_dict(), state)

    def test_legacy_torchscript_metadata_and_predictions(self):
        self.model.eval()
        path = self.base + "_model.pth"
        scripted = torch.jit.trace(self.model, torch.zeros(2, 16, 24, 1), check_trace=False)
        torch.jit.save(scripted, path, _extra_files={"metadata": json.dumps(self.metadata)})
        checkpoint = self.utils.load_checkpoint_from_file(path, "cpu")
        self.assertEqual(checkpoint["metadata"], self.metadata)
        inputs = torch.rand(3, 16, 24, 1)
        with torch.no_grad():
            torch.testing.assert_close(checkpoint["model"](inputs), self.model(inputs))
        restored = self.recognition.apply_checkpoint_to_model(None, checkpoint)
        self.assert_weights_equal(restored.state_dict(), self.model.state_dict())
        for extension in (".pth", ".pt"):
            with self.subTest(extension=extension):
                renamed = self.base + extension
                Path(path).rename(renamed)
                path = renamed
                with mock.patch.object(self.recognition, "get_default_network",
                                       side_effect=AssertionError("Inference reconstructed the architecture")):
                    loaded = json.loads(self.recognition.load_weights(path))
                self.assertEqual(loaded["path"], path)
                self.assertEqual(len(json.loads(self.recognition.find_available_weights(path))), 1)
                self.assert_weights_equal(self.recognition.model.state_dict(), self.model.state_dict())

    def test_legacy_dictionary_with_serialized_model(self):
        path = self.base + "_dict.pth"
        torch.save({"model": self.model, "metadata": self.metadata}, path)
        restored, checkpoint = self.recognition.load_model_from_file(path, "cpu")
        self.assertEqual(checkpoint["metadata"], self.metadata)
        self.assert_weights_equal(restored.state_dict(), self.model.state_dict())

    def test_export_roundtrip_dynamic_batch_and_training_state(self):
        self.model.train()
        self.model.model.bn1.eval()
        modes = [module.training for module in self.model.modules()]
        state = copy.deepcopy(self.model.state_dict())
        expected = copy.deepcopy(self.model).eval()
        path = self.base + "_model.pt2"
        self.utils.save_pytorch_model_as_export(self.model, path, self.metadata)
        self.assertEqual([module.training for module in self.model.modules()], modes)
        self.assert_weights_equal(self.model.state_dict(), state)

        with mock.patch.object(torch.jit, "load", side_effect=AssertionError("Unexpected JIT load")):
            checkpoint = self.utils.load_checkpoint_from_file(path, "cpu")
        self.assertEqual(checkpoint["metadata"], self.metadata)
        self.assert_weights_equal(checkpoint["state_dict"], state)
        exported = checkpoint["model"].eval()
        with torch.no_grad():
            for batch in (1, 3, 7):
                inputs = torch.rand(batch, 16, 24, 1)
                torch.testing.assert_close(exported(inputs), expected(inputs))
        restored = self.recognition.apply_checkpoint_to_model(None, checkpoint)
        restored.train()
        self.assert_weights_equal(restored.state_dict(), state)
        with mock.patch.object(self.recognition, "classes", [0]):
            with self.assertRaises(self.utils.ConfigurationError):
                self.recognition.apply_checkpoint_to_model(None, checkpoint)
        with self.assertRaisesRegex(RuntimeError, "inference-only"):
            exported.train()

    def test_saved_files_discovery_and_export_only_loading(self):
        self.recognition.save_model_files(self.model, self.base, 0.75, epoch=4)
        path = self.base + "_model.pt2"
        self.assertTrue(Path(path).exists())
        self.assertTrue(Path(self.base + "_dict.pth").exists())
        self.assertFalse(Path(self.base + "_model.pth").exists())
        loaded = json.loads(self.recognition.load_weights(self.base))
        self.assertEqual(loaded["path"], path)
        self.assertIsInstance(self.recognition.model, self.utils.ExportedModel)
        available = [json.loads(item) for item in json.loads(self.recognition.find_available_weights())]
        self.assertEqual({item["path"] for item in available}, {self.base + "_dict.pth", path})
        Path(self.base + "_dict.pth").unlink()
        for candidate in (self.base, path):
            with self.subTest(candidate=candidate):
                loaded = json.loads(self.recognition.load_weights(candidate))
                self.assertEqual(loaded["path"], path)
                self.assertEqual(loaded["uniqueness"], 0.75)
                available = json.loads(self.recognition.find_available_weights(candidate))
                self.assertEqual(len(available), 1)
        self.assert_weights_equal(self.recognition.loaded_checkpoint["state_dict"], self.model.state_dict())

    def test_failed_export_loads_new_weights_instead_of_stale_graphs(self):
        base = self.base + "_progress"
        for partial_write in (False, True):
            with self.subTest(partial_write=partial_write):
                self.recognition.save_model_files(self.model, self.base, 0.75, suffix="_progress", epoch=4)
                scripted = torch.jit.trace(copy.deepcopy(self.model).eval(),
                                           torch.zeros(2, 16, 24, 1), check_trace=False)
                torch.jit.save(scripted, base + "_model.pth",
                               _extra_files={"metadata": json.dumps(self.metadata)})
                with torch.no_grad():
                    self.model.model.fc2.bias.add_(1)

                def fail_export(model, path, metadata):
                    if partial_write:
                        Path(path).write_bytes(b"incomplete export")
                    raise RuntimeError("Export failed")

                with mock.patch.object(self.recognition, "save_pytorch_model_as_export",
                                       side_effect=fail_export):
                    self.recognition.save_model_files(self.model, self.base, 0.9, suffix="_progress", epoch=5)

                for candidate in (base, base + "_dict.pth"):
                    loaded = json.loads(self.recognition.load_weights(candidate))
                    self.assertEqual(loaded["path"], base + "_dict.pth")
                    self.assertEqual(loaded["uniqueness"], 0.9)
                    self.assert_weights_equal(self.recognition.model.state_dict(), self.model.state_dict())
                self.assertFalse(Path(base + "_model.pt2").exists())
                self.assertFalse(Path(base + "_model.pth").exists())

    def test_failed_weights_save_preserves_existing_checkpoint(self):
        self.recognition.save_model_files(self.model, self.base, 0.75, epoch=4)
        paths = [Path(self.base + suffix) for suffix in ("_dict.pth", "_model.pt2")]
        saved = [path.read_bytes() for path in paths]
        with torch.no_grad():
            self.model.model.fc2.bias.add_(1)

        def fail_save(checkpoint, path):
            Path(path).write_bytes(b"incomplete checkpoint")
            raise OSError("Disk full")

        with mock.patch.object(torch, "save", side_effect=fail_save), \
                mock.patch.object(self.recognition, "save_pytorch_model_as_export") as export:
            with self.assertRaisesRegex(OSError, "Disk full"):
                self.recognition.save_model_files(self.model, self.base, 0.9, epoch=5)
            export.assert_not_called()
        self.assertEqual([path.read_bytes() for path in paths], saved)

    def test_graph_inference_without_architecture_or_training_imports(self):
        path = self.base + "_model.pt2"
        self.utils.save_pytorch_model_as_export(self.model, path, self.metadata)
        torch.save({"state_dict": self.model.state_dict(), "metadata": self.metadata}, self.base + "_dict.pth")
        images = np.random.default_rng(7).random((5, 16, 24, 1), dtype=np.float32)
        expected = self.recognition.predict_numpy(copy.deepcopy(self.model), images, 2, "cpu")
        utils_spec = importlib.util.spec_from_file_location("trex_utils", RUNTIME_DIR / "trex_utils.py")
        isolated_utils = importlib.util.module_from_spec(utils_spec)
        recognition_spec = importlib.util.spec_from_file_location(
            "isolated_recognition", RUNTIME_DIR / "visual_recognition_torch.py")
        isolated = importlib.util.module_from_spec(recognition_spec)
        with mock.patch.dict(sys.modules, {
            "visual_identification_network_torch": None, "torchvision": None,
            "torchmetrics": None, "trex_utils": isolated_utils,
        }):
            utils_spec.loader.exec_module(isolated_utils)
            recognition_spec.loader.exec_module(isolated)
            for name in ("image_width", "image_height", "image_channels", "classes", "output_path"):
                setattr(isolated, name, getattr(self.recognition, name))
            for candidate in (path, self.base, self.base + "_dict.pth"):
                with self.subTest(candidate=candidate):
                    loaded = json.loads(isolated.load_weights(candidate))
                    self.assertEqual(loaded["path"], path)
                    self.assertIsInstance(isolated.model, isolated_utils.ExportedModel)
                    actual = isolated.predict_numpy(isolated.model, images, 2, "cpu")
                    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)
            available = json.loads(isolated.find_available_weights(self.base))
            self.assertEqual(len(available), 2)
            with mock.patch.multiple(
                isolated, create=True, run_training=False,
                X=list(images[:3]), Y=[0, 1, 2], X_val=list(images[:3]), Y_val=[0, 1, 2],
                max_epochs=1, batch_size=2, learning_rate=0.001, accumulation_step=0,
                global_tracklet=[0, 1], min_iterations=0, verbosity=0,
                filename="checkpoint-test", output_prefix="", best_accuracy_worst_class=0,
                network_version="architecture-no-longer-installed", update_work_percent=None,
            ), mock.patch.object(isolated.ValidationCallback, "evaluate"), \
                    mock.patch.object(isolated.optim, "Adam", side_effect=AssertionError("Inference created an optimizer")):
                loaded = json.loads(isolated.start_learning())
                self.assertEqual(loaded["path"], path)

    def test_continuing_training_reconstructs_before_creating_optimizer(self):
        path = self.base + "_model.pt2"
        self.utils.save_pytorch_model_as_export(self.model, path, self.metadata)
        self.recognition.load_weights(path)
        images = list(np.random.default_rng(2).random((3, 16, 24, 1), dtype=np.float32))

        trained = []
        def train_once(model, train_loader, val_loader, criterion, optimizer, callback, **kwargs):
            self.assertNotIsInstance(model, self.utils.ExportedModel)
            self.assert_weights_equal(model.state_dict(), self.model.state_dict())
            self.assertEqual({id(p) for p in model.parameters()},
                             {id(p) for group in optimizer.param_groups for p in group["params"]})
            before = model.model.fc2.weight.detach().clone()
            model.train()
            optimizer.zero_grad()
            loss = criterion(model(torch.tensor(np.stack(images))), torch.tensor([0, 1, 2]))
            loss.backward()
            optimizer.step()
            self.assertFalse(torch.equal(before, model.model.fc2.weight))
            trained.append(copy.deepcopy(model).eval())
            callback.best_result["unique"] = 0.8
            self.recognition.save_model_files(model, self.base, 0.8, suffix="_progress", epoch=1)

        with mock.patch.multiple(
            self.recognition, create=True, run_training=True,
            X=images, Y=[0, 1, 2], X_val=images, Y_val=[0, 1, 2],
            max_epochs=1, batch_size=3, learning_rate=0.001, accumulation_step=0,
            global_tracklet=[0, 1], min_iterations=0, verbosity=0,
            filename="checkpoint-test", output_prefix="", best_accuracy_worst_class=0,
            do_save_training_images=lambda: False,
        ), mock.patch.object(self.recognition, "train", side_effect=train_once), \
                mock.patch.object(self.recognition.ValidationCallback, "evaluate"):
            loaded = json.loads(self.recognition.start_learning())
        self.assertEqual(loaded["path"], self.base + "_progress_model.pt2")
        self.assertIsInstance(self.recognition.model, self.utils.ExportedModel)
        with torch.no_grad():
            inputs = torch.tensor(np.stack(images))
            torch.testing.assert_close(self.recognition.model(inputs), trained[0](inputs))

    def test_training_cannot_fall_back_to_inference_graph(self):
        path = self.base + "_model.pt2"
        self.utils.save_pytorch_model_as_export(self.model, path, self.metadata)
        with mock.patch.object(self.recognition, "get_default_network",
                               side_effect=ImportError("Architecture unavailable")):
            loaded, _ = self.recognition.load_model_from_file(path, "cpu")
            self.assertIsInstance(loaded, self.utils.ExportedModel)
            with self.assertRaisesRegex(RuntimeError, "Architecture unavailable"):
                self.recognition.load_model_from_file(path, "cpu", for_training=True)

        with mock.patch.object(self.recognition, "get_default_network",
                               return_value=torch.nn.Linear(1, 3)):
            with self.assertRaisesRegex(RuntimeError, "state dict"):
                self.recognition.load_model_from_file(path, "cpu", for_training=True)

    def test_graph_metadata_mismatch_is_rejected(self):
        path = self.base + "_model.pt2"
        self.utils.save_pytorch_model_as_export(self.model, path, self.metadata)
        with mock.patch.object(self.recognition, "classes", [0]):
            with self.assertRaises(self.utils.ConfigurationError):
                self.recognition.load_model_from_file(path, "cpu")


if __name__ == "__main__":
    unittest.main()
