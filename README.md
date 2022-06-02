# Medical Abstract Classification

Para una muestra de la funcionalidad, pueden entrar [acá](http://34.134.234.159:9030).

Para desplegar este archivo localmente, son necesarias las siguientes librerías:
* streamlit==1.8.0
* plotly==5.6.0
* scikit-learn==1.0.2
* contractions==0.1.68
* nltk==3.7

Si quiere ejecutar su archivo localmente, lo puede hacer a través del siguiente comando:

```
streamlit run main.py
```

Para desplegar el contenedor, lo puede ejectutar con los siguientes comandos:

```
docker build -t medical .
docker run -it 9030:9030 --name med_app medical
```
