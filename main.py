import sys
import os
import glob
import warnings
from datetime import datetime
from PyQt5.QtWidgets import (
    QApplication,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QLabel,
    QMessageBox,
    QFileDialog,
    QProgressBar,
)
from PyQt5.QtCore import Qt, QPoint, QThread, pyqtSignal
from PyQt5.QtGui import QPixmap
from dotenv import load_dotenv
from src.uiitems.close_button import CloseButton
from src.assets.smart_background_remover import SmartBackgroundRemover  # Import the new class

# Suppress libpng warnings about incorrect sRGB profiles
warnings.filterwarnings("ignore", ".*iCCP.*")
os.environ['QT_LOGGING_RULES'] = '*.debug=false'

load_dotenv()


def find_resource_path(base_path, filename_pattern):
    """
    Robustly find a resource file by pattern, supporting multiple extensions and locations.

    Args:
        base_path (str): Base directory to search in
        filename_pattern (str): Filename pattern to search for (e.g., "cover", "logo")

    Returns:
        str: Full path to the found file, or None if not found
    """
    # Common image extensions to try
    extensions = [".png", ".jpg", ".jpeg", ".gif", ".bmp", ".ico", ".webp"]

    # First try exact match with different extensions
    for ext in extensions:
        exact_path = os.path.join(base_path, f"{filename_pattern}{ext}")
        if os.path.exists(exact_path):
            return exact_path

    # If exact match fails, try pattern matching
    try:
        # Search for files containing the pattern
        pattern = os.path.join(base_path, f"*{filename_pattern}*")
        matches = glob.glob(pattern)

        # Filter by valid image extensions
        for match in matches:
            if os.path.isfile(match):
                file_ext = os.path.splitext(match)[1].lower()
                if file_ext in extensions:
                    return match
    except Exception:
        pass

    return None


def get_resource_path(resource_type, filename_pattern):
    """
    Get a resource path with fallback locations.

    Args:
        resource_type (str): Type of resource (e.g., "images", "icons", "logos")
        filename_pattern (str): Filename pattern to search for

    Returns:
        str: Full path to the found file, or None if not found
    """
    app_root = get_application_root()

    # Common resource locations to search
    search_paths = [
        os.path.join(app_root, "static", resource_type),
        os.path.join(app_root, "assets", resource_type),
        os.path.join(app_root, "resources", resource_type),
        os.path.join(app_root, resource_type),
        os.path.join(app_root, "static"),  # Fallback for logo images
        # For PyInstaller packaged apps, resources are in the same directory as exe
        os.path.join(app_root, "static", resource_type),
        os.path.join(app_root, "static"),
    ]

    for search_path in search_paths:
        if os.path.exists(search_path):
            found_path = find_resource_path(search_path, filename_pattern)
            if found_path:
                return found_path

    return None


def get_application_root():
    """
    Get the application root directory, handling both development and packaged scenarios.

    Returns:
        str: Path to the application root directory
    """
    if getattr(sys, "frozen", False):
        # Running as compiled executable
        return sys._MEIPASS  # PyInstaller extracts to this temp folder
    else:
        # Running as script
        return os.path.dirname(os.path.abspath(__file__))


class BackgroundRemovalWorker(QThread):
    progress_updated = pyqtSignal(int)
    finished = pyqtSignal()
    
    def __init__(self, folder_path):
        super().__init__()
        self.folder_path = folder_path
        self.remover = SmartBackgroundRemover()
        
    def run(self):
        try:
            # Get all image files in the folder
            image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp']
            image_files = []
            
            for file in os.listdir(self.folder_path):
                if any(file.lower().endswith(ext) for ext in image_extensions):
                    image_files.append(os.path.join(self.folder_path, file))
            
            total_files = len(image_files)
            if total_files == 0:
                self.finished.emit()
                return
            
            # Create output folder with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_folder = os.path.join(self.folder_path, f"no_background_{timestamp}")
            os.makedirs(output_folder, exist_ok=True)
                
            # Process each image with SmartBackgroundRemover
            for i, image_path in enumerate(image_files):
                try:
                    # Create output path
                    base_name = os.path.splitext(os.path.basename(image_path))[0]
                    output_path = os.path.join(output_folder, f"{base_name}_no_bg.png")
                    
                    # Use the improved SmartBackgroundRemover
                    result, mask = self.remover.remove_background(image_path, output_path)
                    
                except Exception as e:
                    print(f"Error processing {image_path}: {e}")
                
                # Update progress
                progress = int((i + 1) / total_files * 100)
                self.progress_updated.emit(progress)
            
            self.finished.emit()
            
        except Exception as e:
            print(f"Background removal error: {e}")
            self.finished.emit()


class MainWorkflowApp(QWidget):
    def __init__(self):
        super().__init__()
        self.init_ui()
        self.img_folder_path = ""
        self.setMouseTracking(True)
        self.oldPos = self.pos()
        self.bg_removal_worker = None

    def init_ui(self):
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setObjectName("App")

        self.setStyleSheet(
            """
            QWidget#App {
                font-family: 'Arial';
                background-color: transparent; 
                border: 2px solid #CDEBF0; 
                border-radius: 20px;
            }
            QPushButton {
                background-color: #87CEEB;
                color: black;
                font-weight: bold;
                border-radius: 8px;
                padding: 10px;
                margin: 10px;
            }
            QPushButton:hover {
                background-color: #BEE0E8;
            }
            QLineEdit {
                border: 2px solid #ccc;
                border-radius: 8px;
                padding: 8px;
                margin: 10px;
                background-color: #BEE0E8;
            }
            QLabel {
                background-color: #CDEBF0;
                color: black;
                font-size: 14px;
                padding: 5px;
                margin: 5px;
                border-radius: 5px;
                border: 1px solid #BEE0E8;
            }
        """
        )

        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(5)
        layout.addLayout(self.create_title_bar())
        layout.addWidget(self.create_logo_label())

        # Background Removal Controls
        layout.addWidget(self.create_bg_removal_controls())

        # After setting up the layout
        self.resize(540, 500)
        self.setLayout(layout)

    def create_button(self, text, slot, style=None):
        button = QPushButton(text, self)
        button.clicked.connect(slot)
        if style:
            button.setStyleSheet(style)
        return button

    def show_custom_message(self):
        msg = QMessageBox(self)
        msg.setIcon(QMessageBox.Information)
        msg.setText("Background removal completed successfully!")
        msg.setWindowTitle("Success")
        msg.setWindowFlags(Qt.FramelessWindowHint | Qt.Dialog | Qt.CustomizeWindowHint)
        msg.setStyleSheet(
            """
        QMessageBox {
            background-color: #BEE0E8;
            color: white;
            font-size: 16px;
        }
        QPushButton {
            color: white;
            border: 2px solid white;
            border-radius: 8px;
            color: white;
            background-color: #BEE0E8;
            padding: 6px;
            font-size: 24px;
            min-width: 70px;
            min-height: 30px;
        }
        QPushButton:hover {
            background-color:  #BEE0E8;
        }
    """
        )
        msg.exec_()

    def create_title_bar(self):
        title_bar = QHBoxLayout()
        close_button = CloseButton(self)
        title_bar.addWidget(close_button, alignment=Qt.AlignRight)
        return title_bar

    def create_logo_label(self):
        logo = QLabel(self)

        # Try to find the cover image with robust path handling
        logo_path = get_resource_path("", "cover")

        if logo_path and os.path.exists(logo_path):
            try:
                # Load the original pixmap without scaling to maintain original size
                pixmap = QPixmap(logo_path)
                if not pixmap.isNull():
                    logo.setPixmap(pixmap)
                    # Set a fixed size for the label to match the image
                    logo.setFixedSize(pixmap.size())
                else:
                    raise ValueError("Failed to load pixmap")
            except Exception as e:
                print(f"Warning: Could not load logo image: {e}")
                # Fallback: create a placeholder with solid background
                logo.setText("Background Removal Tool")
                logo.setStyleSheet(
                    """
                    QLabel {
                        background-color: #CDEBF0;
                        color: black;
                        font-size: 24px;
                        font-weight: bold;
                        border-radius: 10px;
                        padding: 20px;
                        border: 2px solid #BEE0E8;
                    }
                """
                )
            else:
                # Remove default styling for image labels
                logo.setStyleSheet(
                    """
                    QLabel {
                        background-color: transparent;
                        border: none;
                        padding: 0px;
                        margin: 5px;
                    }
                    """
                )
        else:
            # Fallback: create a placeholder with solid background
            logo.setText("Background Removal Tool")
            logo.setStyleSheet(
                """
                QLabel {
                    background-color: #CDEBF0;
                    color: black;
                    font-size: 24px;
                    font-weight: bold;
                    border-radius: 10px;
                    padding: 20px;
                    border: 2px solid #BEE0E8;
                }
            """
            )

        logo.setAlignment(Qt.AlignCenter)
        return logo

    def create_bg_removal_controls(self):
        container = QWidget()
        container.setStyleSheet(
            """
            QWidget {
                background-color: #F0F8FF;
                border-radius: 15px;
                padding: 10px;
                border: 2px solid #CDEBF0;
            }
            """
        )
        layout = QVBoxLayout(container)
        
        # Progress bar for background removal
        self.bg_progress_bar = QProgressBar()
        self.bg_progress_bar.setMinimum(0)
        self.bg_progress_bar.setMaximum(100)
        self.bg_progress_bar.setValue(0)
        self.bg_progress_bar.setTextVisible(True)
        self.bg_progress_bar.setStyleSheet(
            """
            QProgressBar {
                border: 2px solid #CDEBF0;
                border-radius: 8px;
                background: #BEE0E8;
                height: 25px;
                text-align: center;
            }
            QProgressBar::chunk {
                background: #CDEBF0;
                border-radius: 8px;
            }
            """
        )
        self.bg_progress_bar.hide()
        layout.addWidget(self.bg_progress_bar)

        # Folder selection button
        folder_button_style = """
        QPushButton {
            background-color: #E6F3FF;
            color: black;
            font-weight: bold;
            border-radius: 8px;
            padding: 10px;
            margin: 10px;
        }
        QPushButton:hover {
            background-color: #BEE0E8;
        }
        """
        
        self.bg_folder_button = self.create_button(
            "Select Image Folder", self.select_bg_folder_path, folder_button_style
        )
        layout.addWidget(self.bg_folder_button)

        # Start processing button
        cooking_style = """
        QPushButton {
            font-size: 20px; 
            color: #CDEBF0; 
            background-color: black;
            border-radius: 20px;
            padding: 20px;
            font-weight: 900;
            letter-spacing: 2px;
        }
        QPushButton:hover {
            background-color: #333333;
        }
        """
        
        start_button = self.create_button(
            "SUBMIT", self.start_background_removal, cooking_style
        )
        layout.addWidget(start_button)
        
        return container

    def select_bg_folder_path(self):
        folder_path = QFileDialog.getExistingDirectory(self, "Select Folder for Background Removal")
        if folder_path:
            self.img_folder_path = folder_path
            self.bg_folder_button.setText(f"Selected: {os.path.basename(folder_path)}")

    def start_background_removal(self):
        if not self.img_folder_path:
            QMessageBox.warning(
                self,
                "Input Error",
                "Please select an image folder before starting background removal.",
            )
            return

        # Start background removal worker with SmartBackgroundRemover
        self.bg_progress_bar.setValue(0)
        self.bg_progress_bar.show()
        
        self.bg_removal_worker = BackgroundRemovalWorker(self.img_folder_path)
        self.bg_removal_worker.progress_updated.connect(self.bg_progress_bar.setValue)
        self.bg_removal_worker.finished.connect(self.on_bg_removal_finished)
        self.bg_removal_worker.start()

    def on_bg_removal_finished(self):
        self.bg_progress_bar.hide()
        self.show_custom_message()

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.oldPos = event.globalPos()

    def mouseMoveEvent(self, event):
        if event.buttons() == Qt.LeftButton:
            delta = QPoint(event.globalPos() - self.oldPos)
            self.move(self.x() + delta.x(), self.y() + delta.y())
            self.oldPos = event.globalPos()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWorkflowApp()
    window.show()
    sys.exit(app.exec_())