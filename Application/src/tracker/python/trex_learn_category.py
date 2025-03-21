import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, random_split
from torchvision import transforms
from sklearn.metrics import classification_report
import numpy as np
import cv2
import TRex

from trex_utils import clear_caches

categorize = None

class PyTorchModel(nn.Module):
    def __init__(self, input_shape, num_classes):
        super(PyTorchModel, self).__init__()
        self.conv1 = nn.Conv2d(input_shape[0], 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 100, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.batch_norm1 = nn.BatchNorm2d(16)
        self.batch_norm2 = nn.BatchNorm2d(64)
        self.batch_norm3 = nn.BatchNorm2d(100)
        self.dropout = nn.Dropout(0.25)  # Using Dropout instead of Dropout2d
        self.fc1 = nn.Linear(100 * (input_shape[1] // 8) * (input_shape[2] // 8), 100)
        self.fc2 = nn.Linear(100, num_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.pool(F.relu(self.batch_norm1(self.conv1(x))))
        x = self.dropout(x)
        x = self.pool(F.relu(self.batch_norm2(self.conv2(x))))
        x = self.dropout(x)
        x = self.pool(F.relu(self.batch_norm3(self.conv3(x))))
        x = self.dropout(x)
        x = x.reshape(x.size(0), -1)  # Flatten using reshape
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x #return self.softmax(x)

class Categorize:
    def __init__(self, width, height, channels, categories, output_file):
        self.categories_map = eval(categories)
        self.categories = [c for c in self.categories_map]

        self.width = width
        self.height = height
        self.channels = channels

        self.update_required = False
        self.last_size = 0
        self.output_file = output_file

        self.samples = []
        self.labels = []
        self.validation_indexes = np.array([], dtype=int)

        TRex.log("# image dimensions: "+str(self.width)+"x"+str(self.height))
        TRex.log("# initializing categories "+str(categories))

        self.reload_model()

    def plot_samples_grid(self, num_samples_per_class=5):
        """
        Creates a grid of input samples for each class using OpenCV.

        Parameters:
        - num_samples_per_class: The number of samples to display per class.
        """
        # Ensure we have samples to display
        if not self.samples or not self.labels:
            TRex.log("No samples available to display.")
            return

        # Convert samples and labels to numpy arrays for easier indexing
        samples_array = np.array(self.samples)
        labels_array = np.array(self.labels)
        num_classes = len(self.categories)

        # Assume all images have the same size
        img_height, img_width = samples_array[0].shape[:2]

        # Create a blank canvas for the grid
        grid_height = img_height * num_classes
        grid_width = img_width * num_samples_per_class
        grid = np.ones((grid_height, grid_width, 3), dtype=np.uint8) * 255  # White background

        for class_idx, class_name in enumerate(self.categories):
            class_samples = samples_array[labels_array == class_name]

            # If fewer samples are available than requested, adjust
            num_samples = min(num_samples_per_class, len(class_samples))

            for sample_idx in range(num_samples):
                # Get the image
                img = class_samples[sample_idx]
                if len(img.shape) == 2:  # If grayscale, convert to BGR
                    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

                # Determine where to place the image in the grid
                top_left_y = class_idx * img_height
                top_left_x = sample_idx * img_width
                bottom_right_y = top_left_y + img_height
                bottom_right_x = top_left_x + img_width

                # Place the image on the grid
                grid[top_left_y:bottom_right_y, top_left_x:bottom_right_x] = img

            # Put class name on the left of each row
            cv2.putText(grid, f'Class: {class_name}', 
                        (5, (class_idx * img_height) + int(img_height / 2)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)

        # Display the grid image
        TRex.imshow("Sample Images per Class", grid)

    def reload_model(self):
        input_shape = (self.channels, self.height, self.width)
        self.model = PyTorchModel(input_shape, len(self.categories))
        self.model = self.model.float()  # Ensure model is using float
        self.device = TRex.choose_device()
        self.model = self.model.to(self.device)
        TRex.log("Using device: " + str(self.device))
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.0001)
        self.criterion = nn.CrossEntropyLoss()

        TRex.log(str(self.model))

    def send_samples(self):
        TRex.log("# sending "+str(len(self.samples))+" samples")

    def add_images(self, images, labels, force_training):
        prev_L = len(self.labels)
        TRex.log("# previously had "+str(len(self.samples))+" images")

        for image in images:
            self.samples.append(image)

        for l in labels:
            self.labels.append(str(l))

        self.updated_data(prev_L, labels, force_training)

    def updated_data(self, prev_L, labels, force):
        TRex.log("# samples are "+str(np.shape(self.samples))+" labels:"+str(np.shape(self.labels)))
        TRex.log("# "+str(np.unique(self.labels)))

        per_class = {}
        for label in self.labels:
            per_class[label] = 0

        for i in range(len(self.labels)):
            per_class[self.labels[i]] += 1

        missing = int(0.33 * len(self.samples) - len(self.validation_indexes))
        if missing > 0:
            TRex.log("# missing "+str(missing)+" validation samples, adding...")
            next_indexes = np.arange(prev_L, len(labels) + prev_L, dtype=int)
            np.random.shuffle(next_indexes)
            self.validation_indexes = np.concatenate((self.validation_indexes, next_indexes[:missing]), axis=0).astype(int)
            TRex.log("# now have "+str(len(self.validation_indexes))+" validation samples and "+str(len(self.samples) - len(self.validation_indexes))+" training samples")

        TRex.log("# labels dist: "+str(per_class))
        if len(np.unique(self.labels)) == len(self.categories):
            if len(self.samples) - self.last_size >= 500 or force:
                self.update_required = True
                TRex.log("# scheduling update. previous:"+str(self.last_size)+" now:"+str(len(self.samples)))
                self.last_size = len(self.samples)
            else:
                TRex.log("# no update required. previous:"+str(self.last_size)+" now:"+str(len(self.samples)))
        else:
            TRex.log("# not performing training because there are no samples for some categories "+str(per_class))

    def load(self):
        self.reload_model()

        #try:
        if True:
            #with np.load(self.output_file, allow_pickle=True) as npz:
            npz = torch.load(self.output_file, weights_only=False)
            if "samples" in npz:
                # Handle reshaping if data is in a different shape
                if len(np.array(npz["samples"]).shape) == 3:
                    shape = np.shape(npz["samples"])
                    npz["samples"] = np.array(npz["samples"]).reshape((shape[0], shape[1], shape[2], 1))
                shape = np.shape(npz["samples"])

                if shape[1] != self.height or shape[2] != self.width or shape[3] != self.channels:
                    TRex.warn("# loading of weights failed since resolutions differed: " +
                            f"{self.width}x{self.height}x{self.channels} != {shape[2]}x{shape[1]}x{shape[3]}. Change individual_image_size accordingly, or restart the process.")
                    return

                assert shape[1] == self.height and shape[2] == self.width and shape[3] == self.channels
                TRex.log("# loading model with data of shape " + str(shape) +
                        " and current shape " + str(self.height) + "," + str(self.width) + "," + str(self.channels))

                categories_map = npz["categories_map"]#.item()
                TRex.log("# categories_map:" + str(categories_map))
                categories = [c for c in categories_map]

                if categories != self.categories:
                    TRex.log("# categories are different: " +
                            str(categories) + " != " + str(self.categories) + ". replacing current samples.")

                    self.categories = categories
                    self.categories_map = categories_map
                    self.samples = []
                    self.validation_indexes = np.array([], dtype=int)
                    self.labels = []

                m = npz['model_state']#.item(
                self.model.load_state_dict(m)

                validation_indexes = np.array(npz["validation_indexes"]).astype(int)
                #TRex.log("# loading indexes: " + str(validation_indexes))
                TRex.log("# adding data: " + str(np.shape(npz["samples"])))

                # Add current offset to validation_indexes
                validation_indexes += len(self.samples)
                #TRex.log("# with offset: " + str(validation_indexes))
                self.validation_indexes = np.concatenate(
                    (self.validation_indexes, validation_indexes), axis=0)

                # Add data
                prev_L = len(self.labels)
                TRex.log("# unique new labels: " + str(np.unique(npz["labels"])))

                self.samples.extend(npz["samples"])
                self.labels.extend(str(y) for y in npz["labels"])

                self.updated_data(prev_L, [], False)

            if len(self.samples) > 0:
                X = np.array(self.samples).astype(np.float32)

                Y = np.zeros(len(self.labels), dtype=np.float32)
                L = self.categories
                for i in range(len(L)):
                    Y[np.array(self.labels) == L[i]] = self.categories_map[L[i]]
                Y = torch.tensor(Y, dtype=torch.long)

                X_test = torch.tensor(X[self.validation_indexes], dtype=torch.float32) / 127.5 - 1
                Y_test = torch.tensor(Y[self.validation_indexes], dtype=torch.long)
                X_test = X_test.permute(0, 3, 1, 2)  # Change to (batch_size, channels, height, width)
                X_test = X_test.to(self.device)
                Y_test = Y_test.to(self.device)

                self.model.eval()
                with torch.no_grad():
                    outputs = self.model(X_test)
                    loss = self.criterion(outputs, Y_test)
                    TRex.log(f"# Evaluation loss: {loss.item()}")
                    self.update_best_accuracy(X_test, Y_test)
            else:
                TRex.log("# no data available for evaluation")

        #except Exception as e:
        #    TRex.warn(f"Loading weights failed: {str(e)}")

    def perform_training(self):
        TRex.log("# performing training...")
        batch_size = 32

        Y = np.zeros(len(self.labels), dtype=int)
        L = self.categories

        for i in range(len(L)):
            Y[np.array(self.labels) == L[i]] = self.categories_map[L[i]]

        TRex.log("Y = "+str(Y))
        Y = torch.tensor(Y, dtype=torch.long)
        #Y = torch.tensor([self.categories_map[label] for label in self.labels], dtype=torch.long)
        X = torch.tensor(np.array(self.samples), dtype=torch.float32) / 127.5 - 1

        training_indexes = np.arange(len(X), dtype=int)
        TRex.log("training:"+str(np.shape(training_indexes))+" val:"+str(np.shape(self.validation_indexes)))
        TRex.log("#/class: "+str(np.unique(Y, return_counts=True)))
        training_indexes = np.delete(training_indexes, self.validation_indexes)

        #self.plot_samples_grid()

        X = X.permute(0, 3, 1, 2)  # Change to (batch_size, channels, height, width)

        train_len = int(len(X) * 0.77)
        X_train = X[training_indexes]
        Y_train = Y[training_indexes]

        X_test = X[self.validation_indexes]
        Y_test = Y[self.validation_indexes]

        TRex.log("# training samples: "+str(len(X_train))+" validation samples: "+str(len(X_test)))
        TRex.log("# training samples / class: "+str(np.unique(Y_train.cpu().numpy(), return_counts=True)))
        TRex.log("# validation samples / class: "+str(np.unique(Y_test.cpu().numpy(), return_counts=True)))

        train_dataset = TensorDataset(X_train, Y_train)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        self.model.train()
        for epoch in range(10):
            running_loss = 0.0
            for inputs, labels in train_loader:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                inputs, labels = inputs.float(), labels.long()

                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()

            TRex.log(f"Epoch {epoch+1}, Loss: {running_loss/len(train_loader)}")

        self.model.eval()
        with torch.no_grad():
            X_test = X_test.to(self.device)
            Y_test = Y_test.to(self.device)
            outputs = self.model(X_test)
            loss = self.criterion(outputs, Y_test)
            TRex.log(f"# Final validation loss: {loss.item()}")
            self.update_best_accuracy(X_test, Y_test)

        try:
            torch.save({
                'model_state': self.model.state_dict(),
                'optimizer_state': self.optimizer.state_dict(),
                'samples': self.samples,
                'labels': self.labels,
                'validation_indexes': self.validation_indexes,
                'categories_map': self.categories_map,
                'categories': self.categories
            }, self.output_file)
            TRex.log("# UPDATE: saved samples and weights.")
        except Exception as e:
            TRex.log("Saving weights and samples failed: "+str(e))

    def update_best_accuracy(self, X_test, Y_test):
        with torch.no_grad():
            TRex.log("# evaluating model...")
            TRex.log("# X_test: "+str(X_test.shape)+" Y_test: "+str(Y_test.shape))
            TRex.log("#ytest/class: "+str(np.unique(Y_test.cpu().numpy(), return_counts=True)))

            y_pred = self.model(X_test)
            softmax = nn.Softmax(dim=1)
            y_pred = softmax(self.model(X_test)).argmax(dim=1)

            #TRex.log("# y_pred: "+str(y_pred.shape)+" y_pred:"+str(y_pred))
            TRex.log("#ypred/class: "+str(np.unique(y_pred.cpu().numpy(), return_counts=True)))

            report = classification_report(Y_test.cpu().numpy(), y_pred.cpu().numpy(), output_dict=True, zero_division=0)

        for key in report:
            TRex.log("report: "+str(key)+" "+str(report[key]))
        TRex.log(str(report))
        set_best_accuracy(float(report["accuracy"]))

    def update(self):
        if self.update_required:
            self.update_required = False
            self.perform_training()

    def predict(self, images):
        assert self.model
        images = torch.tensor(np.array(images), dtype=torch.float32) / 127.5 - 1
        images = images.permute(0, 3, 1, 2)  # Change to (batch_size, channels, height, width)
        images = images.to(self.device)
        with torch.no_grad():
            y = self.model(images)
            y = nn.Softmax(dim=1)(y).argmax(dim=1).cpu().numpy().tolist()
        return y


def start():
    global categorize, categories, width, height, channels, output_file

    if type(categorize) == type(None) or categorize.categories != categories:
        categorize = Categorize(width, height, channels, categories, output_file)

    TRex.log("# initialized with categories"+str(categories)+".")

def load():
    global categorize, width, height, channels, categories
    assert type(categorize) != type(None)

    if type(categorize) == type(None) and width == categorize.width and height == categorize.height and channels == categorize.channels and eval(categories) == categorize.categories_map:
        start()
    else:
        TRex.log("# model already exists. reloading model")
        categorize.load()

def add_images():
    global categorize, additional, additional_labels, force_training
    assert type(categorize) != type(None)

    TRex.log("# adding "+str(len(additional))+" images (force:"+str(force_training)+")")
    categorize.add_images(additional, additional_labels, force_training)

    del additional
    del additional_labels

def post_queue():
    global categorize
    assert type(categorize) != type(None)

    categorize.update()

def send_samples():
    global categorize
    categorize.send_samples()

def predict():
    global categorize, images, receive
    assert type(categorize) != type(None)

    results = categorize.predict(images)
    del images
    receive(results)

def clear():
    clear_caches()
    TRex.log("# cleared images")

def clear_images():
    global categorize

    TRex.log("# clearing images")
    categorize = None
    clear()
    start()
