import os
import shutil
import sys
import torch
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QLabel, QFileDialog, QMessageBox, QListWidget, \
    QListWidgetItem, QCheckBox, QVBoxLayout, QScrollArea, QWidget, QFrame
from PyQt5.QtGui import QPixmap, QIcon
from PyQt5.QtCore import Qt, QSize
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
 
#set-up program GUI 
class NotificationBox(QFrame):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(300, 100)
        self.setStyleSheet("background-color: #444; color: white; border-radius: 5px; padding: 10px;")
        self.layout = QVBoxLayout(self)
        self.layout.setAlignment(Qt.AlignTop)
        self.setLayout(self.layout)
        self.message_label = QLabel("", self)
        self.message_label.setStyleSheet("font-size: 14px; color: white; padding: 5px;")
        self.message_label.setWordWrap(True)
        self.layout.addWidget(self.message_label)
        self.closeButton = QPushButton("Close", self)
        self.closeButton.setStyleSheet("""
            background-color: #222; 
            color: white;
            font-weight: bold;
            border-radius: 5px;
        """)
        self.closeButton.clicked.connect(self.closeNotifications)
        self.layout.addWidget(self.closeButton)
        self.hide()
 
    def addNotification(self, message, parent_frame):
        self.message_label.setText(message)
        self.adjustSizeBasedOnContent()
        if parent_frame:
            self.centerInFrame(parent_frame)
        self.show()
 
    def adjustSizeBasedOnContent(self):
        message_height = self.message_label.sizeHint().height()
        total_height = message_height + self.closeButton.sizeHint().height() + 30
        self.setFixedSize(300, total_height)
 
    def centerInFrame(self, parent_frame):
        frame_geometry = parent_frame.geometry()
        x = frame_geometry.x() + (frame_geometry.width() - self.width()) // 2
        y = frame_geometry.y() + (frame_geometry.height() - self.height()) // 2
        self.move(x, y)
 
    def closeNotifications(self):
        self.hide()
 
#Image Tagging Algorithm 
class ImageTagger(QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI()
        self.model, self.processor = self.load_model()
        self.save_folder = None
        self.uploaded_images = []
        self.image_tags = {}
        self.all_tags = set()
        self.notificationBox = NotificationBox(self)
 
    def load_model(self):
        model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        return model, processor
 
    def initUI(self):
        self.setWindowTitle('Campus Event Photo Organizer')
        self.setGeometry(100, 100, 1200, 800)
        self.setStyleSheet("background-color: #2F2F2F; color: white;")
        self.titleLabel = QLabel("Campus Event Photo Organizer", self)
        self.titleLabel.setGeometry(50, 10, 1100, 40)
        self.titleLabel.setAlignment(Qt.AlignCenter)
        self.titleLabel.setStyleSheet("font-size: 24px; font-weight: bold; font-family: 'Times New Roman'; color: white;")
        self.imageFrame = QFrame(self)
        self.imageFrame.setGeometry(50, 60, 900, 500)
        self.imageFrame.setStyleSheet("border: 3px solid #444; border-radius: 10px;")
        self.label = QLabel(self.imageFrame)
        self.label.setGeometry(0, 0, 900, 500)
        self.label.setAlignment(Qt.AlignCenter)
        self.thumbnailList = QListWidget(self)
        self.thumbnailList.setGeometry(50, 600, 900, 150)
        self.thumbnailList.setStyleSheet("""
            background-color: #3A3A3A;
            border-radius: 10px;
            padding: 5px;
            outline: none;
        """)
        self.thumbnailList.setIconSize(QSize(100, 100))
        self.thumbnailList.itemClicked.connect(self.displayImageFromThumbnail)
        button_style = """
            QPushButton {
                background-color: black; 
                color: white; 
                font-weight: bold; 
                font-family: 'Times New Roman'; 
                font-size: 18px;
                border: 2px solid #444;
                border-radius: 10px;
            }
            QPushButton:hover {
                background-color: #444; 
                color: #FFF;
            }
            QPushButton:pressed {
                background-color: #222; 
                color: #FFF;
                border: 2px solid #FFF;
            }
        """
        self.uploadButton = QPushButton('Upload Photos', self)
        self.uploadButton.setGeometry(1000, 50, 150, 50)
        self.uploadButton.setStyleSheet(button_style)
        self.uploadButton.clicked.connect(self.uploadImages)
        self.saveFolderButton = QPushButton('Set Save Folder', self)
        self.saveFolderButton.setGeometry(1000, 150, 150, 50)
        self.saveFolderButton.setStyleSheet(button_style)
        self.saveFolderButton.clicked.connect(self.setSaveFolder)
        self.removeButton = QPushButton('Remove Photo', self)
        self.removeButton.setGeometry(1000, 250, 150, 50)
        self.removeButton.setStyleSheet(button_style)
        self.removeButton.clicked.connect(self.removeImage)
        self.tagFilterWidget = QScrollArea(self)
        self.tagFilterWidget.setGeometry(1000, 350, 150, 350)
        self.tagFilterWidget.setStyleSheet("background-color: #3A3A3A; border-radius: 10px; padding: 5px;")
        self.tagFilterWidget.setWidgetResizable(True)
        self.tagFilterContainer = QWidget()
        self.tagFilterLayout = QVBoxLayout(self.tagFilterContainer)
        self.tagFilterLayout.setAlignment(Qt.AlignTop)
        self.tagFilterWidget.setWidget(self.tagFilterContainer)
        self.tagCheckBoxes = {}
 
        self.show()
 
    #Get/Select images to use
    def uploadImages(self):
        options = QFileDialog.Options()
        fileNames, _ = QFileDialog.getOpenFileNames(self, 'Open Image Files', '', 'All Files (*)', options=options)
        if fileNames:
            for fileName in fileNames:
                self.uploaded_images.append(fileName)
                self.addThumbnail(fileName)
                self.processImage(fileName)
            self.notificationBox.addNotification(f"{len(fileNames)} images uploaded successfully!", self.imageFrame)
 
    #Select image to present on GUI
    def addThumbnail(self, fileName):
        image = QPixmap(fileName)
        if image.isNull():
            print(f"Failed to load image: {fileName}")
            return
        image = image.scaled(100, 100, Qt.KeepAspectRatio)
        icon = QIcon(image)
        item = QListWidgetItem()
        item.setIcon(icon)
        item.setText(os.path.basename(fileName))
        self.thumbnailList.addItem(item)
 
    #Display main selected image on DUI
    def displayImageFromThumbnail(self, item):
        fileName = None
        for path in self.uploaded_images:
            if os.path.basename(path) == item.text():
                fileName = path
                break
        if fileName and os.path.exists(fileName):
            self.displayImage(fileName)
 
    #Display other images on GUI
    def displayImage(self, fileName):
        image = QPixmap(fileName)
        if not image.isNull():
            self.label.setPixmap(image.scaled(self.label.size(), Qt.KeepAspectRatio))
 
    #Manually select which folder to save in
    def setSaveFolder(self):
        folder = QFileDialog.getExistingDirectory(self, 'Select Folder')
        if folder:
            self.save_folder = folder
            QMessageBox.information(self, 'Folder Selected', f"Images will be saved to: {self.save_folder}")
 
    #Analyze and Process images
    def processImage(self, fileName):
        tags = self.generateTags(fileName)
        self.image_tags[fileName] = tags
        self.saveImage(fileName, tags)
        self.updateTagFilterCheckBoxes(tags)
 
    #Tag images based on processed information
    def generateTags(self, fileName):
        image = Image.open(fileName)
        candidate_labels = [
            "event", "campus", "group", "lecture", "classroom", "student", "university", "people",
            "celebration", "athlete", "sport", "professor", "device", "technology"
        ]
        inputs = self.processor(text=candidate_labels, images=image, return_tensors="pt", padding=True)
        with torch.no_grad():
            outputs = self.model(**inputs)
        logits_per_image = outputs.logits_per_image.squeeze()
        probs = logits_per_image.softmax(dim=0)
        confidence_threshold = 0.3
        return [candidate_labels[i] for i, prob in enumerate(probs.tolist()) if prob >= confidence_threshold]
 
    #Save images to selected folder based on tags
    def saveImage(self, fileName, tags):
        if not self.save_folder:
            self.notificationBox.addNotification("Please set a save folder first!", self.imageFrame)
            return
        try:
            base_name = os.path.basename(fileName)
            for tag in tags:
                tag_folder = os.path.join(self.save_folder, tag)
                os.makedirs(tag_folder, exist_ok=True)
                dest_path = os.path.join(tag_folder, base_name)
                shutil.copy(fileName, dest_path)
            self.notificationBox.addNotification(f"Image '{base_name}' has been saved successfully.", self.imageFrame)
        except Exception as e:
            self.notificationBox.addNotification(f"An error occurred while saving the image: {str(e)}", self.imageFrame)
 
    #Adjust Tag to view
    def updateTagFilterCheckBoxes(self, tags):
        for tag in tags:
            if tag not in self.tagCheckBoxes:
                checkbox = QCheckBox(tag)
                checkbox.setStyleSheet("color: white; font-size: 14px;")
                checkbox.stateChanged.connect(self.filterImagesByTag)
                self.tagCheckBoxes[tag] = checkbox
                self.tagFilterLayout.addWidget(checkbox)
 
    #Select tag and view related images
    def filterImagesByTag(self):
        selected_tags = [tag for tag, checkbox in self.tagCheckBoxes.items() if checkbox.isChecked()]
        self.thumbnailList.clear()
        for fileName, tags in self.image_tags.items():
            if set(selected_tags).intersection(tags) or not selected_tags:
                self.addThumbnail(fileName)
 
    #Remove image from folder
    def removeImage(self):
        selected_items = self.thumbnailList.selectedItems()
        if not selected_items:
            QMessageBox.warning(self, 'No Selection', "Please select an image to remove.")
            return
        selected_item = selected_items[0]
        fileName = None
        for path in self.uploaded_images:
            if os.path.basename(path) == selected_item.text():
                fileName = path
                break
        if fileName:
            tags = self.image_tags.get(fileName, [])
            for tag in tags:
                tag_folder = os.path.join(self.save_folder, tag)
                if os.path.exists(tag_folder):
                    image_path_in_tag_folder = os.path.join(tag_folder, os.path.basename(fileName))
                    if os.path.exists(image_path_in_tag_folder):
                        os.remove(image_path_in_tag_folder)
            self.uploaded_images.remove(fileName)
            self.thumbnailList.takeItem(self.thumbnailList.row(selected_item))
            QMessageBox.information(self, 'Image Removed', f"Image '{fileName}' removed from sorted folders.")
 
#Main Program running 
if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = ImageTagger()
    sys.exit(app.exec_())
