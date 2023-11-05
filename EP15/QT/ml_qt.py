# Librerías diseño
from PyQt5.QtWidgets import QMainWindow, QApplication, QGraphicsDropShadowEffect
from PyQt5.uic import loadUi
from PyQt5.QtCore import QPoint
from PyQt5 import QtCore, QtWidgets
from PyQt5.QtGui import QColor, QIntValidator
import sys
from imagenes import imagenes
# Librerías data
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix

class Formulario(QMainWindow):
    def __init__(self):
        super(Formulario, self).__init__()
        self.rf = None
        loadUi("caries.ui", self)

        self.scaler = MinMaxScaler()
        self.train_model()

        self.bt_normal.hide()
        self.click_posicion = QPoint()
        self.bt_minimize.clicked.connect(lambda :self.showMinimized())
        self.bt_normal.clicked.connect(self.control_bt_normal)
        self.bt_maximize.clicked.connect(self.control_bt_maximize)
        self.bt_close.clicked.connect(lambda: self.close())

        # Eliminar barra de título y opacidad
        self.setWindowFlag(QtCore.Qt.FramelessWindowHint)
        self.setWindowOpacity(1)
        self.setWindowFlags(QtCore.Qt.FramelessWindowHint)
        self.setAttribute(QtCore.Qt.WA_TranslucentBackground)

        # SizeGrip
        self.gripSize = 10
        self.grip = QtWidgets.QSizeGrip(self)
        self.grip.resize(self.gripSize, self.gripSize)
        # Mover ventana
        self.frame_superior.mouseMoveEvent = self.mover_ventana

        # Botones
        self.bt_limpiar.clicked.connect(self.clear_data)
        self.bt_guardar.clicked.connect(self.save_data)

        self.shadow_frame(self.frame_datos)
        self.shadow_frame(self.frame_resultado)

        self.edad.setValidator(QIntValidator())
        self.altura.setValidator(QIntValidator())
        self.peso.setValidator(QIntValidator())
        self.cintura.setValidator(QIntValidator())
        self.vista_izquierda.setValidator(QIntValidator())
        self.vista_derecha.setValidator(QIntValidator())
        self.audicion_izquierda.setValidator(QIntValidator())
        self.audicion_derecha.setValidator(QIntValidator())
        self.sistolica.setValidator(QIntValidator())
        self.presion_arterial.setValidator(QIntValidator())
        self.glucosa_en_ayunas.setValidator(QIntValidator())
        self.colesterol.setValidator(QIntValidator())
        self.triglicerido.setValidator(QIntValidator())
        self.hdl.setValidator(QIntValidator())
        self.ldl.setValidator(QIntValidator())
        self.hemoglobina.setValidator(QIntValidator())
        self.proteina_urinaria.setValidator(QIntValidator())
        self.creatinina_serica.setValidator(QIntValidator())
        self.ast.setValidator(QIntValidator())
        self.alt.setValidator(QIntValidator())
        self.gtp.setValidator(QIntValidator())

    def train_model(self):
        df = pd.read_csv('dataset_caries.csv')

        x = df.drop(['caries'], axis=1)
        y = df['caries']

        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=42)

        self.scaler.fit(x_train)

        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(x_train, y_train)

        self.rf = rf

        y_pred = rf.predict(x_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f'Accuracy del modelo: {accuracy}\n')

        matrix = confusion_matrix(y_test, y_pred)
        print('MATRIZ DE CONFUSION:\n')
        print(matrix)

    # Funciones
    def shadow_frame(self, frame):
        shadow = QGraphicsDropShadowEffect(self)
        shadow.setBlurRadius(10)
        shadow.setOffset(5,5)
        shadow.setColor(QColor(127, 90, 240, 255))
        frame.setGraphicsEffect(shadow)

    def clear_data(self):
        self.nombre.clear()
        self.edad.clear()
        self.altura.clear()
        self.peso.clear()
        self.cintura.clear()
        self.vista_izquierda.clear()
        self.vista_derecha.clear()
        self.audicion_izquierda.clear()
        self.audicion_derecha.clear()
        self.sistolica.clear()
        self.presion_arterial.clear()
        self.glucosa_en_ayunas.clear()
        self.colesterol.clear()
        self.triglicerido.clear()
        self.hdl.clear()
        self.ldl.clear()
        self.hemoglobina.clear()
        self.proteina_urinaria.clear()
        self.creatinina_serica.clear()
        self.ast.clear()
        self.alt.clear()
        self.gtp.clear()

    def save_data(self):
        try:
            nombre = self.nombre.text()
            edad = self.edad.text()
            altura = self.altura.text()
            peso = self.peso.text()
            cintura = self.cintura.text()
            vista_izquierda = self.vista_izquierda.text()
            vista_derecha = self.vista_derecha.text()
            audicion_izquierda = self.audicion_izquierda.text()
            audicion_derecha = self.audicion_derecha.text()
            sistolica = self.sistolica.text()
            presion_arterial = self.presion_arterial.text()
            glucosa_en_ayunas = self.glucosa_en_ayunas.text()
            colesterol = self.colesterol.text()
            triglicerido = self.triglicerido.text()
            hdl = self.hdl.text()
            ldl = self.ldl.text()
            hemoglobina = self.hemoglobina.text()
            proteina_urinaria = self.proteina_urinaria.text()
            creatinina_serica = self.creatinina_serica.text()
            ast = self.ast.text()
            alt = self.alt.text()
            gtp = self.gtp.text()

            data = [edad, altura, peso, cintura, vista_izquierda, vista_derecha, audicion_izquierda, audicion_derecha,
                    sistolica, presion_arterial, glucosa_en_ayunas, colesterol, triglicerido, hdl, ldl, hemoglobina,
                    proteina_urinaria, creatinina_serica, ast, alt, gtp]
            data = [float(x) for x in data]

            data_scaled = self.scaler.transform([data])
            if self.rf is not None:
                caries = self.rf.predict(data_scaled)

            # Mostrar el resultado en la interfaz gráfica
            if caries == 0:
                self.resultado.setText(f"Paciente: {nombre}\n\n¡NO eres propenso a padecer caries dental!\n\nModelo Random Forest")
            elif caries == 1:
                self.resultado.setText(f"Paciente: {nombre}\n\nSI eres propenso a padecer caries dental.\n\nModelo Random Forest")

        except Exception as e:
            error_message = str(e)
            QtWidgets.QMessageBox.critical(self, "Error", f"Error al guardar los datos: {error_message}")

    def control_bt_normal(self):
        self.showNormal()
        self.bt_normal.hide()
        self.bt_maximize.show()

    def control_bt_maximize(self):
        self.showMaximized()
        self.bt_maximize.hide()
        self.bt_normal.show()

    # SizeGrip 2
    def resizeEvent(self, event):
        rect = self.rect()
        self.grip.move(rect.right() - self.gripSize, rect.bottom() - self.gripSize)
    # Mover ventana 2
    def mousePressEvent(self, event):
        self.click_posicion = event.globalPos()

    def mover_ventana(self, event):
        if self.isMaximized() == False:
            if event.buttons() == QtCore.Qt.LeftButton:
                self.move(self.pos() + event.globalPos() - self.click_posicion)
                self.click_posicion = event.globalPos()
                event.accept
        if event.globalPos().y() <=5 or event.globalPos().x() <=5:
            self.showMaximized()
            self.bt_maximize.hide()
            self.bt_normal.show()
        else:
            self.showNormal()
            self.bt_normal.hide()
            self.bt_maximize.show()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    my_app = Formulario()
    my_app.show()
    sys.exit(app.exec())