"""
app.py — Entry point khởi động UI.
Chạy: python app.py
"""

import sys
from PyQt6.QtWidgets import QApplication
from User_interface.main_window import MainWindow


def main():
    app = QApplication(sys.argv)
    app.setApplicationName("Football Analysis System")

    window = MainWindow()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()