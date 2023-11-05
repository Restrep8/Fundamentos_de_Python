# Librerias
import tkinter as tk
import pandas as pd
import numpy as np
from tkinter import font
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix
from tkinter import messagebox

# Cargar los datos
df = pd.read_csv('dataset_caries.csv')

# Dividir conjunto de datos en variables de entrada (X) y salida (Y)
x = df.drop(['caries'], axis=1)
y = df['caries']

# Dividir datos de entrada y salida en conjuntos de entrenamiento y prueba
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=42)

# Normalización de datos
scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# Ajustar modelo Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(x_train, y_train)

# Predecir valores de prueba
y_pred = rf.predict(x_test)

# Matriz de confusión
matrix = confusion_matrix(y_test, y_pred)
print('MATRIZ DE CONFUSION:\n')
print(matrix)

y_pred = rf.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

def borrar_campos():
    nombre_entry.delete(0, tk.END)
    edad_entry.delete(0, tk.END)
    altura_entry.delete(0, tk.END)
    peso_entry.delete(0, tk.END)
    cintura_entry.delete(0, tk.END)
    vista_izquierda_entry.delete(0, tk.END)
    vista_derecha_entry.delete(0, tk.END)
    audicion_izquierda_entry.delete(0, tk.END)
    audicion_derecha_entry.delete(0, tk.END)
    sistolica_entry.delete(0, tk.END)
    presion_arterial_entry.delete(0, tk.END)
    glucosa_en_ayunas_entry.delete(0, tk.END)
    colesterol_entry.delete(0, tk.END)
    triglicerido_entry.delete(0, tk.END)
    hdl_entry.delete(0, tk.END)
    ldl_entry.delete(0, tk.END)
    hemoglobina_entry.delete(0, tk.END)
    proteina_urinaria_entry.delete(0, tk.END)
    creatinina_serica_entry.delete(0, tk.END)
    ast_entry.delete(0, tk.END)
    alt_entry.delete(0, tk.END)
    gtp_entry.delete(0, tk.END)
    diagnostico_label.config(text="")

def obtener_diagnostico():
    nombre = nombre_entry.get()
    edad = float(edad_entry.get())
    altura = float(altura_entry.get())
    peso = float(peso_entry.get())
    cintura = float(cintura_entry.get())
    vista_izquierda = float(vista_izquierda_entry.get())
    vista_derecha = float(vista_derecha_entry.get())
    audicion_izquierda = float(audicion_izquierda_entry.get())
    audicion_derecha = float(audicion_derecha_entry.get())
    sistolica = float(sistolica_entry.get())
    presion_arterial = float(presion_arterial_entry.get())
    glucosa_en_ayunas = float(glucosa_en_ayunas_entry.get())
    colesterol = float(colesterol_entry.get())
    triglicerido = float(triglicerido_entry.get())
    hdl = float(hdl_entry.get())
    ldl = float(ldl_entry.get())
    hemoglobina = float(hemoglobina_entry.get())
    proteina_urinaria = float(proteina_urinaria_entry.get())
    creatinina_serica = float(creatinina_serica_entry.get())
    ast = float(ast_entry.get())
    alt = float(alt_entry.get())
    gtp = float(gtp_entry.get())

    # Realizar la predicción y mostrar el diagnóstico
    data = [[edad, altura, peso, cintura, vista_izquierda, vista_derecha, audicion_izquierda, audicion_derecha, sistolica, presion_arterial, glucosa_en_ayunas, colesterol, triglicerido, hdl, ldl, hemoglobina, proteina_urinaria, creatinina_serica, ast, alt, gtp]]

    # Realizar la predicción
    data_scaled = scaler.transform(np.array(data))
    caries = rf.predict(data_scaled)

    # Mostrar el diagnóstico
    if caries == 0:
        diagnostico_label.config(text=f"Nombre completo: {nombre}\n\nNuestro diagnóstico sugiere que el paciente NO es propenso a padecer caries dental.\nRecomendamos seguir las indicaciones de su médico y realizar chequeos regulares para monitorear cualquier cambio en su salud.\n\nEste modelo utiliza el clasificador Random Forest.\n\nAccuracy: {accuracy:.2%}\nPrecisión: {precision:.2%}\nRecall: {recall:.2%}\nF1-Score: {f1:.2%}\n\nRecuerde que estos resultados son basados en los datos proporcionados y pueden estar sujetos a variaciones.\n")
    elif caries == 1:
        diagnostico_label.config(text=f"Nombre completo: {nombre}\n\nNuestro diagnóstico sugiere que el paciente SI es propenso a padecer caries dental.\nEs importante buscar atención médica especializada lo antes posible para iniciar un tratamiento adecuado.\n\nEste modelo utiliza el clasificador Random Forest.\n\nAccuracy: {accuracy:.2%}\nPrecisión: {precision:.2%}\nRecall: {recall:.2%}\nF1-Score: {f1:.2%}\n\nRecuerde que estos resultados son basados en los datos proporcionados y pueden estar sujetos a variaciones.\n")

def mostrar_instrucciones():
    mensaje = "¡Hola! Esto es un test para predecir si eres propenso a padecer caries dental.\n\nLa caries dental es una enfermedad bucal que afecta a los dientes. El humo y el uso del tabaco en diferentes formas son factores de riesgo conocido para el desarrollo de esta. Provocando problemas como sequedad bucal y cambios en la composición de la saliva.\n\nIngrese a continuación cada uno de los datos que se piden, con el fin de conocer su diagnóstico.\n\n"
    messagebox.showinfo("Ayuda", mensaje)

def salir():
    window.destroy()

window = tk.Tk()
window.title("¿FUMAS? - PREDICE SI ERES PROPENSO A PADECER CARIES DENTAL")

# Frame principal
frame = tk.Frame(window)
frame.grid(pady=10)
frame.configure(background='aquamarine')

# Cambiar el color de fondo
window.configure(background='aquamarine')

# Frame para el formulario
datos_frame = tk.Frame(frame)
datos_frame.grid(row=0, column=0, padx=10)
datos_frame.configure(background='mediumpurple1')

# Crear una fuente con un tamaño más grande
titulo_font = font.Font(size=14, weight='bold')

# Crear una etiqueta con el título y aplicar la fuente
titulo_datos_label = tk.Label(datos_frame, text="     Formulario", font=titulo_font)
titulo_datos_label.grid(row=0, column=1, sticky='w')
titulo_datos_label.configure(background='mediumpurple1')

# Columna 1
nombre_label = tk.Label(datos_frame, text="Nombre completo:")
nombre_label.grid(row=1, column=0, sticky="e")

nombre_entry = tk.Entry(datos_frame)
nombre_entry.grid(row=1, column=1, padx=10, pady=5)

edad_label = tk.Label(datos_frame, text="Edad:")
edad_label.grid(row=2, column=0, sticky="e")

edad_entry = tk.Entry(datos_frame)
edad_entry.grid(row=2, column=1, padx=10, pady=5)

altura_label = tk.Label(datos_frame, text="Altura:")
altura_label.grid(row=3, column=0, sticky="e")

altura_entry = tk.Entry(datos_frame)
altura_entry.grid(row=3, column=1, padx=10, pady=5)

peso_label = tk.Label(datos_frame, text="Peso:")
peso_label.grid(row=4, column=0, sticky="e")

peso_entry = tk.Entry(datos_frame)
peso_entry.grid(row=4, column=1, padx=10, pady=5)

cintura_label = tk.Label(datos_frame, text="Cintura:")
cintura_label.grid(row=5, column=0, sticky="e")

cintura_entry = tk.Entry(datos_frame)
cintura_entry.grid(row=5, column=1, padx=10, pady=5)

vista_izquierda_label = tk.Label(datos_frame, text="Vista izquierda:")
vista_izquierda_label.grid(row=6, column=0, sticky="e")

vista_izquierda_entry = tk.Entry(datos_frame)
vista_izquierda_entry.grid(row=6, column=1, padx=10, pady=5)

vista_derecha_label = tk.Label(datos_frame, text="Vista derecha:")
vista_derecha_label.grid(row=7, column=0, sticky="e")

vista_derecha_entry = tk.Entry(datos_frame)
vista_derecha_entry.grid(row=7, column=1, padx=10, pady=5)

audicion_izquierda_label = tk.Label(datos_frame, text="Audicion izquierda:")
audicion_izquierda_label.grid(row=8, column=0, sticky="e")

audicion_izquierda_entry = tk.Entry(datos_frame)
audicion_izquierda_entry.grid(row=8, column=1, padx=10, pady=5)

audicion_derecha_label = tk.Label(datos_frame, text="Audicion derecha:")
audicion_derecha_label.grid(row=9, column=0, sticky="e")

audicion_derecha_entry = tk.Entry(datos_frame)
audicion_derecha_entry.grid(row=9, column=1, padx=10, pady=5)

sistolica_label = tk.Label(datos_frame, text="Sistólica:")
sistolica_label.grid(row=10, column=0, sticky="e")

sistolica_entry = tk.Entry(datos_frame)
sistolica_entry.grid(row=10, column=1, padx=10, pady=5)

presion_arterial_label = tk.Label(datos_frame, text="Presión arterial:")
presion_arterial_label.grid(row=11, column=0, sticky="e")

presion_arterial_entry = tk.Entry(datos_frame)
presion_arterial_entry.grid(row=11, column=1, padx=10, pady=5)

# Columna 2
glucosa_en_ayunas_label = tk.Label(datos_frame, text="Glucosa en ayunas:")
glucosa_en_ayunas_label.grid(row=1, column=2, sticky="e")

glucosa_en_ayunas_entry = tk.Entry(datos_frame)
glucosa_en_ayunas_entry.grid(row=1, column=3, padx=10, pady=5)

colesterol_label = tk.Label(datos_frame, text="Colesterol:")
colesterol_label.grid(row=2, column=2, sticky="e")

colesterol_entry = tk.Entry(datos_frame)
colesterol_entry.grid(row=2, column=3, padx=10, pady=5)

triglicerido_label = tk.Label(datos_frame, text="Triglicérido:")
triglicerido_label.grid(row=3, column=2, sticky="e")

triglicerido_entry = tk.Entry(datos_frame)
triglicerido_entry.grid(row=3, column=3, padx=10, pady=5)

hdl_label = tk.Label(datos_frame, text="HDL:")
hdl_label.grid(row=4, column=2, sticky="e")

hdl_entry = tk.Entry(datos_frame)
hdl_entry.grid(row=4, column=3, padx=10, pady=5)

ldl_label = tk.Label(datos_frame, text="LDL:")
ldl_label.grid(row=5, column=2, sticky="e")

ldl_entry = tk.Entry(datos_frame)
ldl_entry.grid(row=5, column=3, padx=10, pady=5)

hemoglobina_label = tk.Label(datos_frame, text="Hemoglobina:")
hemoglobina_label.grid(row=6, column=2, sticky="e")

hemoglobina_entry = tk.Entry(datos_frame)
hemoglobina_entry.grid(row=6, column=3, padx=10, pady=5)

proteina_urinaria_label = tk.Label(datos_frame, text="Proteína urinaria:")
proteina_urinaria_label.grid(row=7, column=2, sticky="e")

proteina_urinaria_entry = tk.Entry(datos_frame)
proteina_urinaria_entry.grid(row=7, column=3, padx=10, pady=5)

creatinina_serica_label = tk.Label(datos_frame, text="Creatinina sérica:")
creatinina_serica_label.grid(row=8, column=2, sticky="e")

creatinina_serica_entry = tk.Entry(datos_frame)
creatinina_serica_entry.grid(row=8, column=3, padx=10, pady=5)

ast_label = tk.Label(datos_frame, text="AST:")
ast_label.grid(row=9, column=2, sticky="e")

ast_entry = tk.Entry(datos_frame)
ast_entry.grid(row=9, column=3, padx=10, pady=5)

alt_label = tk.Label(datos_frame, text="ALT:")
alt_label.grid(row=10, column=2, sticky="e")

alt_entry = tk.Entry(datos_frame)
alt_entry.grid(row=10, column=3, padx=10, pady=5)

gtp_label = tk.Label(datos_frame, text="GTP:")
gtp_label.grid(row=11, column=2, sticky="e")

gtp_entry = tk.Entry(datos_frame)
gtp_entry.grid(row=11, column=3, padx=10, pady=5)

# Columna 1
nombre_label.configure(background='mediumpurple1')
edad_label.configure(background='mediumpurple1')
altura_label.configure(background='mediumpurple1')
peso_label.configure(background='mediumpurple1')
cintura_label.configure(background='mediumpurple1')
vista_izquierda_label.configure(background='mediumpurple1')
vista_derecha_label.configure(background='mediumpurple1')
audicion_izquierda_label.configure(background='mediumpurple1')
audicion_derecha_label.configure(background='mediumpurple1')
sistolica_label.configure(background='mediumpurple1')
presion_arterial_label.configure(background='mediumpurple1')

# Columna 2
glucosa_en_ayunas_label.configure(background='mediumpurple1')
colesterol_label.configure(background='mediumpurple1')
triglicerido_label.configure(background='mediumpurple1')
hdl_label.configure(background='mediumpurple1')
ldl_label.configure(background='mediumpurple1')
hemoglobina_label.configure(background='mediumpurple1')
proteina_urinaria_label.configure(background='mediumpurple1')
creatinina_serica_label.configure(background='mediumpurple1')
ast_label.configure(background='mediumpurple1')
alt_label.configure(background='mediumpurple1')
gtp_label.configure(background='mediumpurple1')

# Frame para el diagnóstico
diagnostico_frame = tk.Frame(frame)
diagnostico_frame.grid(row=0, column=1, padx=10)
diagnostico_frame.configure(background='aquamarine')

# Crear una etiqueta con el título y aplicar la fuente
titulo_diagnostico_label = tk.Label(diagnostico_frame, text="Diagnóstico", font=titulo_font)
titulo_diagnostico_label.grid(row=0, column=0, sticky='w')
titulo_diagnostico_label.configure(background='aquamarine')

diagnostico_label = tk.Label(diagnostico_frame, text="", wraplength=200, justify=tk.LEFT)
diagnostico_label.grid(row=1, column=0, padx=10, pady=10)
diagnostico_label.configure(background='aquamarine')

# Ajustar el tamaño de los títulos para que estén a la misma altura
titulo_datos_label.grid(row=0, column=1, sticky='w')
titulo_diagnostico_label.grid(row=0, column=0, sticky='n')

# Frame para los botones
botones_frame = tk.Frame(window)
botones_frame.grid(pady=10)
botones_frame.configure(background='aquamarine')

borrar_button = tk.Button(botones_frame, text="Limpiar", command=borrar_campos)
borrar_button.grid(row=0, column=0, padx=10)

diagnostico_button = tk.Button(botones_frame, text="Guardar", command=obtener_diagnostico)
diagnostico_button.grid(row=0, column=1, padx=10)

salir_button = tk.Button(botones_frame, text="Salir", command=salir)
salir_button.grid(row=0, column=2, padx=10)

instrucciones_button = tk.Button(botones_frame, text="Ayuda", command=mostrar_instrucciones)
instrucciones_button.grid(row=0, column=3, padx=10)

window.mainloop()