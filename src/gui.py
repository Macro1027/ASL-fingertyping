import sys
import cv2
import mediapipe as mp
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QVBoxLayout, QWidget, QColorDialog, QLabel, QHBoxLayout
from PyQt5.QtGui import QPainter, QColor, QPen, QImage, QPixmap
from PyQt5.QtCore import Qt, QPoint, QTimer

class Drawing:
    def __init__(self):
        self.lines = []  # Store the current lines drawn
        self.history = []  # Store history for undo/redo
        self.redo_stack = []  # Stack for redo actions

    def add_line(self, line):
        self.lines.append(line)
        self.history.append(line)  # Save to history
        self.redo_stack.clear()  # Clear redo stack on new action

    def undo(self):
        if self.history:
            last_line = self.history.pop()  # Get the last drawn line
            self.redo_stack.append(last_line)  # Save it for redo
            self.lines.remove(last_line)  # Remove from current lines

    def redo(self):
        if self.redo_stack:
            last_redo_line = self.redo_stack.pop()  # Get the last undone line
            self.lines.append(last_redo_line)  # Add to current lines
            self.history.append(last_redo_line)  # Add back to history

    def clear_canvas(self):
        self.lines.clear()  # Clear all lines
        self.history.clear()  # Clear history
        self.redo_stack.clear()  # Clear redo stack

class GestureRecognition:
    def __init__(self):
        self.gesture = None

    def recognize_gesture(self, hand_landmarks):
        if hand_landmarks:
            index_finger_tip = hand_landmarks.landmark[mp.solutions.hands.HandLandmark.INDEX_FINGER_TIP].y
            middle_finger_tip = hand_landmarks.landmark[mp.solutions.hands.HandLandmark.MIDDLE_FINGER_TIP].y
            thumb_tip = hand_landmarks.landmark[mp.solutions.hands.HandLandmark.THUMB_TIP].x
            index_finger_tip_x = hand_landmarks.landmark[mp.solutions.hands.HandLandmark.INDEX_FINGER_TIP].x

            if index_finger_tip < middle_finger_tip:
                self.gesture = 'undo'
            elif middle_finger_tip < index_finger_tip:
                self.gesture = 'redo'
            elif thumb_tip < index_finger_tip_x:
                self.gesture = 'clear'

        return self.gesture

class DrawingArea(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setGeometry(0, 0, 800, 600)
        self.setFocusPolicy(Qt.StrongFocus)

        self.drawing_color = QColor(0, 0, 0)
        self.line_width = 2
        self.drawing = Drawing()
        self.current_cursor_position = None

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)  # Enable anti-aliasing for smoother lines
        pen = QPen()
        pen.setWidth(self.line_width)
        pen.setColor(self.drawing_color)
        painter.setPen(pen)

        for line in self.drawing.lines:
            painter.drawLine(line[0], line[1])

        if self.current_cursor_position is not None:
            cursor_pen = QPen(QColor(255, 0, 0))
            cursor_pen.setWidth(3)
            painter.setPen(cursor_pen)
            painter.drawEllipse(self.current_cursor_position, 5, 5)

    def change_pen_color(self, color):
        self.drawing_color = color

    def clear_canvas(self):
        self.drawing.clear_canvas()
        self.update()

class WhiteboardApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Whiteboard App")
        self.setGeometry(100, 100, 1400, 600)

        self.drawing_area = DrawingArea(self)
        
        self.cap = cv2.VideoCapture(0)
        self.camera_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.camera_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        self.new_camera_width = 320
        self.new_camera_height = 240

        self.camera_label = QLabel(self)
        self.camera_label.setFixedSize(self.new_camera_width, self.new_camera_height)

        self.color_button = QPushButton("Change Color")
        self.color_button.clicked.connect(self.change_pen_color)

        self.clear_button = QPushButton("Clear Canvas")
        self.clear_button.clicked.connect(self.clear_canvas)

        layout = QVBoxLayout()
        layout.addWidget(self.color_button)
        layout.addWidget(self.clear_button)
        layout.addWidget(self.drawing_area)

        container = QWidget()
        container.setLayout(layout)

        main_layout = QHBoxLayout()
        main_layout.addWidget(container)
        main_layout.addWidget(self.camera_label)

        main_widget = QWidget()
        main_widget.setLayout(main_layout)
        self.setCentralWidget(main_widget)

        self.setStyleSheet("""
            QMainWindow {
                background-color: #2E3440;
            }
            QPushButton {
                background-color: #4C566A;
                color: #ECEFF4;
                border-radius: 10px;
                padding: 10px;
            }
            QPushButton:hover {
                background-color: #5E81AC;
            }
            QLabel {
                background-color: #3B4252;
                border-radius: 10px;
            }
            QWidget {
                background-color: #2E3440;
            }
        """)

        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)

        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands()
        self.mp_draw = mp.solutions.drawing_utils

        self.last_finger_position = None
        self.gesture_recognition = GestureRecognition()

    def change_pen_color(self):
        color = QColorDialog.getColor()
        if color.isValid():
            self.drawing_area.change_pen_color(color)

    def clear_canvas(self):
        self.drawing_area.clear_canvas()

    def update_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            return

        frame = cv2.flip(frame, 1)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(frame_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                self.mp_draw.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                gesture = self.gesture_recognition.recognize_gesture(hand_landmarks)

                if gesture == 'undo':
                    self.drawing_area.drawing.undo()
                elif gesture == 'redo':
                    self.drawing_area.drawing.redo()
                elif gesture == 'clear':
                    self.drawing_area.clear_canvas()

                index_finger_tip = hand_landmarks.landmark[mp.solutions.hands.HandLandmark.INDEX_FINGER_TIP]
                index_finger_dip = hand_landmarks.landmark[mp.solutions.hands.HandLandmark.INDEX_FINGER_DIP]

                finger_x = int(index_finger_tip.x * self.camera_width)
                finger_y = int(index_finger_tip.y * self.camera_height)
                canvas_x, canvas_y = self.map_finger_position_to_canvas(finger_x, finger_y)

                self.drawing_area.current_cursor_position = QPoint(canvas_x, canvas_y)

                if index_finger_tip.y < index_finger_dip.y:
                    if self.last_finger_position is not None:
                        self.drawing_area.drawing.add_line((self.last_finger_position, QPoint(canvas_x, canvas_y)))
                    self.last_finger_position = QPoint(canvas_x, canvas_y)
                else:
                    self.last_finger_position = None

                self.drawing_area.update()

        resized_frame = cv2.resize(frame_rgb, (self.new_camera_width, self.new_camera_height))
        image = QImage(resized_frame, resized_frame.shape[1], resized_frame.shape[0], QImage.Format_RGB888)
        self.camera_label.setPixmap(QPixmap.fromImage(image))

    def map_finger_position_to_canvas(self, x, y):
        normalized_x = x / self.camera_width
        normalized_y = y / self.camera_height
        canvas_x = int(normalized_x * self.drawing_area.width())
        canvas_y = int(normalized_y * self.drawing_area.height())
        return canvas_x, canvas_y

    def closeEvent(self, event):
        self.cap.release()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = WhiteboardApp()
    window.show()
    sys.exit(app.exec_())