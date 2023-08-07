# MLOps-Azure, creando una canalización de un modelo a produccion

[Machine Learning Operations (MLOps)](https://learn.microsoft.com/en-us/azure/machine-learning/concept-model-management-and-deployment?view=azureml-api-2) se basa en los principios y prácticas de DevOps que aumentan la eficiencia en el despliegue y el flujo de trabajo en modelos de Machine Learning en producción.

Este repositorio contiene el código y las pautas para configurar el flujo de trabajo de MLOps con Azure:
Se trata de un modelo de clasificación que ayuda a los profesionales a detectar posibles casos de diabetes teniendo en cuenta ciertos datos tomados del paciente.
 
El objetivo es crear una canalizacion end2end para hacer las predicciones en tiempo real cuando se introduzcan los datos del paciente, y poder re-entrenar el modelo utilizando scripts cuando se presenten una cantidad considerable de nuevos datos o cuando el modelo empiece a degradarse.

Si bien los Jupyter notebooks son excelentes para explorar y compartir datos, generalmente se consideran que **"No son para produccion"** debido a las dificultades de integración con la arquitectura de un software y a la naturaleza de investigacion por celdas de los notebooks, además que no son adecuados para implementarlos en un endpoint, como una API y mantenerla con las mejores prácticas de DevOps. De cara a estas complicaciones debemos convertir estos cuadernos en scripts, y hacer el codigo modular, para que sea facilmente utilizable y automatizable, este proyecto es un ejemplo sencillo de cómo podría ser la ejecución de un modelo 'en producción' utilizando Azure Cloud. Mediante la incorporación de cuatro pilares de ingeniería de software sólida:

- Control de versiones
- Código modular
- Pruebas unitarias y de integración
- CI/CD

## Pronto mas detalles.



