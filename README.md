## APP Flask ACC
Esta es la app para poner en producción el modelo entranado en AECC para la detección de triaje. La app funciona con Flask

## Flask
Flask is a popular lightweight Python web MVC framework. Compared to Django, Flask based projects are usually simpler to implement and easier to maintain. 

## Paquetes
En la máquina donde se lance el servicio se tienen que instalar estos paqutes de python (pip)  
Tensorflow-1.14.0.dist-info
Flask-1.0.3.dist-info
h5py-2.9.0.dist-info
numpy-1.16.4.dist-info

## Modelo 
En este repositorio esta el último modelo entrenado y el actual en producción

## Servicio 
python3 app.py creara un servicio http que solo admite una petición via post y que se configura desde el Drupal
https://www.aecc.es/es/admin/config/services/tensorflow/settings
