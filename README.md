# TC5061 — Código de experimentación base con MLflow

## 1. Descripción del proyecto

Este proyecto implementa un modelo de Machine Learning basado en regresión logística para clasificar clientes que podrían cancelar un servicio (*Customer Churn*), utilizando el conjunto de datos Telco Customer Churn.

Se utiliza **MLflow Tracking** para registrar los parámetros de configuración del modelo, las métricas de evaluación y los artefactos generados en cada ejecución. Esto permite comparar diferentes configuraciones experimentales y facilitar la reproducibilidad de los resultados.

## 2. Estructura del repositorio

```text
TC5061-semana-04/
├── train.py
├── requirements.txt
└── README.md
```

El conjunto de datos `WA_Fn-UseC_-Telco-Customer-Churn.csv` debe estar disponible en la ruta esperada por el script de entrenamiento.

**Descripción de los archivos:**

- `train.py`: script de entrenamiento y evaluación del modelo de Machine Learning.
- `requirements.txt`: archivo con las dependencias de Python necesarias para ejecutar el proyecto.
- `README.md`: documentación e instrucciones de ejecución.

## 3. Requisitos previos

Para ejecutar el proyecto se requiere:

- Python 3.12.3.
- pip para instalar las dependencias.
- El conjunto de datos Telco Customer Churn.
- Las dependencias especificadas en `requirements.txt`.

## 4. Instalación

### 4.1. Clonar el repositorio

Ejecuta los siguientes comandos en una terminal:

```bash
git clone https://github.com/iplatamx/TC5061-semana-04.git
cd TC5061-semana-04
```

### 4.2. Instalar las dependencias

Instala las bibliotecas necesarias mediante el archivo `requirements.txt`:

```bash
python -m pip install -r requirements.txt
```

### 4.3. Preparar el conjunto de datos

Asegúrate de que el archivo `WA_Fn-UseC_-Telco-Customer-Churn.csv` se encuentre en la ruta que utiliza el script `train.py`.

Si el archivo no está incluido en el repositorio, deberás obtenerlo por separado y colocarlo en la ubicación esperada por el script.

## 5. Ejecución del entrenamiento

### 5.1. Iniciar el servidor de MLflow

Abre una terminal y ejecuta:

```bash
mlflow server --host 127.0.0.1 --port 5000
```

Mantén esta terminal abierta mientras realizas los experimentos.

### 5.2. Ejecutar el script de entrenamiento

Abre una segunda terminal, accede al directorio del proyecto y ejecuta:

```bash
python train.py --c 1.0 --solver lbfgs --max-iter 1000 --random-state 42
```

El script entrena el modelo de regresión logística, evalúa su desempeño y registra los resultados en MLflow.

### 5.3. Parámetros de ejecución

El script admite los siguientes argumentos:

| Parámetro | Descripción | Valor predeterminado |
|---|---|---|
| `--c` | Parámetro de regularización inversa de la regresión logística. | `1.0` |
| `--solver` | Algoritmo de optimización utilizado por el modelo. | `lbfgs` |
| `--max-iter` | Número máximo de iteraciones del algoritmo. | `1000` |
| `--random-state` | Semilla aleatoria utilizada para favorecer la reproducibilidad. | `42` |

Ejemplo de ejecución con una configuración diferente:

```bash
python train.py --c 0.25 --solver lbfgs --max-iter 1000 --random-state 7
```

## 6. Seguimiento de experimentos con MLflow

Con el servidor activo, accede a la interfaz de MLflow desde el navegador:

http://127.0.0.1:5000

El nombre del experimento registrado es:

`Telco_Customer_Churn_Baseline`

MLflow permite consultar y comparar las diferentes ejecuciones del modelo.

En cada ejecución se registran los siguientes elementos:

- **Parámetros:** configuración utilizada para entrenar el modelo.
- **Métricas:** accuracy, precision, recall, F1-score y ROC-AUC.
- **Artefactos:** archivo `evaluation.txt` y modelo entrenado.

Estos registros permiten analizar el desempeño de las diferentes configuraciones experimentales y consultar los resultados de cada ejecución.

## 7. Reproducibilidad de los experimentos

Para comprobar la reproducibilidad, ejecuta dos veces el mismo comando utilizando los mismos parámetros y la misma semilla aleatoria:

```bash
python train.py --c 1.0 --solver lbfgs --max-iter 1000 --random-state 42
```

En las pruebas realizadas, ambas ejecuciones produjeron los mismos resultados:

| Métrica | Resultado |
|---|---:|
| Accuracy | 0.8055 |
| Precision | 0.6572 |
| Recall | 0.5588 |
| F1-score | 0.6040 |
| ROC-AUC | 0.8421 |

La coincidencia de estos resultados demuestra que, en las ejecuciones realizadas bajo las mismas condiciones, se obtuvo el mismo desempeño del modelo.

## 8. Autor

**Iván Plata Garnica**  
**Matrícula:** A01840199

**Asignatura:** TC5061 — Operaciones de aprendizaje automático  
**Institución:** Tecnológico de Monterrey

## 9. Repositorio individual

El código fuente, el script de entrenamiento y el archivo de dependencias se encuentran en el siguiente repositorio de GitHub:

https://github.com/iplatamx/TC5061-semana-04
