import sys
import sqlite3
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QAction, QTabWidget, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QLineEdit, QTextEdit, QPushButton, QTableWidget, QTableWidgetItem, QMessageBox, QDialog, QFormLayout
)
from PyQt5.QtCore import Qt
from predictor import Predictor
import numpy as np
from datetime import datetime

# --- User Login Dialog ---
class LoginDialog(QDialog):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('User Login')
        self.username = ''
        layout = QFormLayout()
        self.user_input = QLineEdit()
        layout.addRow('Username:', self.user_input)
        login_btn = QPushButton('Login')
        login_btn.clicked.connect(self.accept)
        layout.addWidget(login_btn)
        self.setLayout(layout)
    def accept(self):
        self.username = self.user_input.text().strip()
        if self.username:
            super().accept()
        else:
            QMessageBox.warning(self, 'Error', 'Please enter a username.')

# --- Logger Tab ---
class LoggerTab(QWidget):
    def __init__(self, db_conn, username):
        super().__init__()
        self.conn = db_conn
        self.username = username
        layout = QVBoxLayout()
        self.log_input = QTextEdit()
        self.time_input = QLineEdit()
        self.time_input.setPlaceholderText('Time and date (ISO format, e.g., 2025-06-13T10:00)')
        self.log_btn = QPushButton('Log Entry')
        self.log_btn.clicked.connect(self.log_entry)
        layout.addWidget(QLabel('How are you feeling or what are you doing?'))
        layout.addWidget(self.log_input)
        layout.addWidget(self.time_input)
        layout.addWidget(self.log_btn)
        self.setLayout(layout)
    def log_entry(self):
        user_log = self.log_input.toPlainText().strip()
        user_time = self.time_input.text().strip()
        if not user_log or not user_time:
            QMessageBox.warning(self, 'Error', 'Please enter log and time/date.')
            return
        from predictor import Predictor
        predictor = Predictor(None, [], [])
        keywords, polarity, subjectivity, action = predictor.extract_features_from_text(user_log)
        c = self.conn.cursor()
        c.execute('''INSERT INTO logs (timestamp, user_input, keywords, polarity, subjectivity, target_action, user_time, username) VALUES (?, ?, ?, ?, ?, ?, ?, ?)''',
                  (datetime.now().isoformat(), user_log, keywords, polarity, subjectivity, action, user_time, self.username))
        self.conn.commit()
        QMessageBox.information(self, 'Success', f'Logged! Extracted action: {action}')
        self.log_input.clear()
        self.time_input.clear()

# --- Prediction Tab ---
class PredictionTab(QWidget):
    def __init__(self, db_conn, username):
        super().__init__()
        self.conn = db_conn
        self.username = username
        layout = QVBoxLayout()
        self.query_input = QTextEdit()
        self.time_input = QLineEdit()
        self.time_input.setPlaceholderText('Time and date (ISO format, e.g., 2025-06-13T23:00)')
        self.predict_btn = QPushButton('Predict')
        self.predict_btn.clicked.connect(self.predict_action)
        self.result_label = QLabel('')
        layout.addWidget(QLabel('Ask a prediction question (e.g., What is the chance of Jack crying over his breakup in the next 1 hour?)'))
        layout.addWidget(self.query_input)
        layout.addWidget(self.time_input)
        layout.addWidget(self.predict_btn)
        layout.addWidget(self.result_label)
        self.setLayout(layout)
    def predict_action(self):
        query = self.query_input.toPlainText().strip()
        pred_time = self.time_input.text().strip()
        if not query or not pred_time:
            QMessageBox.warning(self, 'Error', 'Please enter a query and time/date.')
            return
        c = self.conn.cursor()
        c.execute('SELECT user_input, keywords, polarity, subjectivity, target_action, user_time FROM logs WHERE username=?', (self.username,))
        data = c.fetchall()
        if len(data) < 5:
            self.result_label.setText('Need at least 5 logs to train the model.')
            return
        all_keywords = sorted({k for row in data for k in row[1].split(',') if k.strip()})
        actions = list(set([row[4] for row in data if row[4]]))
        predictor = Predictor(None, all_keywords, actions)
        X, y = predictor.prepare_training_data(data)
        from tensorflow import keras
        from tensorflow.keras import layers
        model = keras.Sequential([
            layers.Input(shape=(X.shape[1],)),
            layers.Dense(32, activation='relu'),
            layers.Dense(1, activation='sigmoid')
        ])
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        model.fit(X, y, epochs=30, verbose=0)
        predictor.model = model
        prob, action = predictor.predict(query, pred_time)
        self.result_label.setText(f'Prediction for "{action}" at {pred_time}: {prob}%')

# --- Database Viewer Tab ---
class DBViewerTab(QWidget):
    def __init__(self, db_conn, username):
        super().__init__()
        self.conn = db_conn
        self.username = username
        layout = QVBoxLayout()
        self.table = QTableWidget()
        self.refresh_btn = QPushButton('Refresh')
        self.refresh_btn.clicked.connect(self.load_data)
        layout.addWidget(self.table)
        layout.addWidget(self.refresh_btn)
        self.setLayout(layout)
        self.load_data()
    def load_data(self):
        c = self.conn.cursor()
        c.execute('SELECT timestamp, user_input, keywords, polarity, subjectivity, target_action, user_time FROM logs WHERE username=? ORDER BY id DESC', (self.username,))
        rows = c.fetchall()
        self.table.setRowCount(len(rows))
        self.table.setColumnCount(7)
        self.table.setHorizontalHeaderLabels(['Timestamp', 'Input', 'Keywords', 'Polarity', 'Subjectivity', 'Action', 'Time'])
        for i, row in enumerate(rows):
            for j, val in enumerate(row):
                self.table.setItem(i, j, QTableWidgetItem(str(val)))

# --- Main Window ---
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Behavior Prediction Engine Dashboard')
        self.conn = sqlite3.connect('behavior.db')
        self.ensure_schema()
        self.username = ''
        self.initUI()
    def ensure_schema(self):
        c = self.conn.cursor()
        c.execute('''CREATE TABLE IF NOT EXISTS logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            user_input TEXT,
            keywords TEXT,
            polarity REAL,
            subjectivity REAL,
            target_action TEXT,
            user_time TEXT,
            username TEXT
        )''')
        self.conn.commit()
    def initUI(self):
        # Menu bar
        menubar = self.menuBar()
        userMenu = menubar.addMenu('User')
        loginAction = QAction('Login', self)
        loginAction.triggered.connect(self.login)
        userMenu.addAction(loginAction)
        # Tabs
        self.tabs = QTabWidget()
        self.setCentralWidget(self.tabs)
        self.show_login()
    def show_login(self):
        login = LoginDialog()
        if login.exec_() == QDialog.Accepted:
            self.username = login.username
            self.tabs.clear()
            self.tabs.addTab(LoggerTab(self.conn, self.username), 'Logger')
            self.tabs.addTab(PredictionTab(self.conn, self.username), 'Prediction')
            self.tabs.addTab(DBViewerTab(self.conn, self.username), 'Database Viewer')
    def login(self):
        self.show_login()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
