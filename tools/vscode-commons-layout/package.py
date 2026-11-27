"""Package the extension as a single, replaceable commons-layout.vsix."""

from pathlib import Path
import json
import re
import xml.etree.ElementTree as ET
import zipfile


root = Path(__file__).resolve().parent
package_text = (root / "package.json").read_text()
package = json.loads(package_text)
major, minor, patch = map(int, package["version"].split("."))
package["version"] = f"{major}.{minor}.{patch + 1}"
package_text = re.sub(r'("version"\s*:\s*")[^"]+(")',
    lambda match: match[1] + package["version"] + match[2], package_text, count=1)
(root / "package.json").write_text(package_text)
files = ["package.json", "extension.js", "tokens.js", "fields.js", "README.md", "LICENSE"]
output = root / "commons-layout.vsix"
temporary = root / "commons-layout.vsix.tmp"

manifest = ET.Element("PackageManifest", {
    "xmlns": "http://schemas.microsoft.com/developer/vsx-schema/2011", "Version": "2.0.0"
})
metadata = ET.SubElement(manifest, "Metadata")
ET.SubElement(metadata, "Identity", {"Language": "en-US", "Id": package["name"],
    "Version": package["version"], "Publisher": package["publisher"]})
ET.SubElement(metadata, "DisplayName").text = package["displayName"]
ET.SubElement(metadata, "Description", {"{http://www.w3.org/XML/1998/namespace}space": "preserve"}).text = package["description"]
ET.SubElement(metadata, "Tags").text = ",".join(package["keywords"])
ET.SubElement(metadata, "Categories").text = ",".join(package["categories"])
properties = ET.SubElement(metadata, "Properties")
ET.SubElement(properties, "Property", {"Id": "Microsoft.VisualStudio.Code.Engine", "Value": package["engines"]["vscode"]})
ET.SubElement(properties, "Property", {"Id": "Microsoft.VisualStudio.Code.ExtensionKind", "Value": "workspace"})
ET.SubElement(ET.SubElement(manifest, "Installation"), "InstallationTarget", {"Id": "Microsoft.VisualStudio.Code"})
ET.SubElement(manifest, "Dependencies")
assets = ET.SubElement(manifest, "Assets")
for asset_type, name in [("Microsoft.VisualStudio.Code.Manifest", "package.json"),
                         ("Microsoft.VisualStudio.Services.Content.Details", "README.md"),
                         ("Microsoft.VisualStudio.Services.Content.License", "LICENSE")]:
    ET.SubElement(assets, "Asset", {"Type": asset_type, "Path": "extension/" + name, "Addressable": "true"})

content_types = ET.Element("Types", {"xmlns": "http://schemas.openxmlformats.org/package/2006/content-types"})
for extension, content_type in [("json", "application/json"), ("js", "application/javascript"),
                                ("md", "text/markdown"), ("vsixmanifest", "text/xml")]:
    ET.SubElement(content_types, "Default", {"Extension": extension, "ContentType": content_type})
ET.SubElement(content_types, "Override", {"PartName": "/extension/LICENSE", "ContentType": "text/plain"})

with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED) as archive:
    archive.writestr("extension.vsixmanifest", ET.tostring(manifest, encoding="utf-8", xml_declaration=True))
    archive.writestr("[Content_Types].xml", ET.tostring(content_types, encoding="utf-8", xml_declaration=True))
    for name in files:
        archive.write(root / name, "extension/" + name)
with zipfile.ZipFile(temporary) as archive:
    assert archive.testzip() is None
    for name in files:
        assert archive.read("extension/" + name) == (root / name).read_bytes()
temporary.replace(output)

for previous in root.glob("commons-layout-*.vsix"):
    if re.fullmatch(r"commons-layout-\d+\.\d+\.\d+\.vsix", previous.name):
        previous.unlink()
print(f"Packaged and verified {output.name}, internal version {package['version']} ({output.stat().st_size} bytes)")
