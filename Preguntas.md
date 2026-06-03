
# Preguntas sobre el ejemplo de clasificación de imágenes con PyTorch y MLP

## 1. Dataset y Preprocesamiento
- ¿Por qué es necesario redimensionar las imágenes a un tamaño fijo para una MLP?

La necesidad de redimensionar las imágenes proviene de que la cantidad de entradas de una red MLP está determinada y no puede cambiarse, por lo que si se ingresa con mayor cantidad de entradas, el modelo no va a responder correctamente. Para redimensionar el tamaño se escribe:

```python
input_size = 64*64*3
nn.Linear(input_size, 512)
```
Esta línea define la cantidad de entradas para imágenes de 64 x 64 en formato RGB. Si la entrada es distinta, se debe redimensionar con la entrada esperada.

- ¿Qué ventajas ofrece Albumentations frente a otras librerías de transformación como `torchvision.transforms`?

Ambas librerías están diseñadas para realizar cambios en las imágenes, como rotación, cambio de brillo, entre otros. Las principales diferencias entre estas radican en su funcionamiento y su desarrollo:

En cuanto al desarrollo, Albumentations está desarrollada en C++, mientras que `torchvision.transforms` está desarrollada en Python, lo que marca una diferencia significativa en velocidad, ya que C++ es capaz de procesar los cálculos de forma más rápida.

Por otra parte, en cuanto al funcionamiento, Albumentations es capaz de transformar simultáneamente la imagen y sus anotaciones asociadas (como cajas de delimitación para detección de objetos o máscaras de píxeles para segmentación) en una sola operación. A diferencia de `torchvision.transforms`, donde si rotás una imagen, tenés que programar manualmente la transformación correspondiente para rotar también las coordenadas de su etiqueta. Albumentations asegura que cualquier cambio geométrico se aplique de forma idéntica y sincronizada a todo el conjunto de datos, lo cual evita errores de desalineación en el entrenamiento, y optimiza el flujo de trabajo, ya que no requerie código extra para procesar las etiquetas asociadas.

- ¿Qué hace `A.Normalize()`? ¿Por qué es importante antes de entrenar una red?

Cuando los datos de entrada tienen escalas muy distintas o valores absolutos grandes, la superficie de pérdida se vuelve extremadamente alargada y asimétrica. 

Al calcular los gradientes en una superficie asimétrica, el optimizador, como gradient-descent, empieza a oscilar abruptamente de un lado al otro de los extremos, en lugar de avanzar directo hacia el mínimo. Esto causa dos problemas:

1. El entrenamiento se vuelve mucho más lento, ya que se desperdician pasos en corregir oscilaciones.

2. Corrés el riesgo de que los gradientes exploten o se desvanezcan, haciendo que la red directamente no aprenda.

Al aplicar `A.Normalize()`, se modifica la distribución de los datos para que todas las variables tengan la misma escala (media 0 y varianza 1). Esto redondea y simetriza la superficie de pérdida. Los pasos se vuelven más estables y directos hacia el mínimo global. El resultado es un entrenamiento más rápido, matemáticamente estable, y con una convergencia mucho más eficiente.


- ¿Por qué convertimos las imágenes a `ToTensorV2()` al final de la pipeline?

Las librerías de carga de imágenes trabajan con arrays de NumPy y en formato HWC (Alto, Ancho, Canales). PyTorch no usa eso, necesita Tensores de PyTorch y en formato CHW (Canales, Alto, Ancho). ToTensorV2() hace ese cambio de formato y de tipo de dato justo al final para que la GPU pueda procesarlo.

## 2. Arquitectura del Modelo
- ¿Por qué usamos una red MLP en lugar de una CNN aquí? ¿Qué limitaciones tiene?

Se elige una arquitectura MLP en esta instancia inicial porque su implementación es mucho más sencilla y directa para establecer un modelo base (baseline).

Sin embargo, las limitaciones de la MLP aparecen rápidamente al trabajar con datos visuales debido a dos factores:

1. Las capas están completamente conectadas (fully connected), cada píxel de entrada se conecta con cada neurona de la siguiente capa. Si usamos imágenes de alta resolución, la multiplicación de los píxeles de entrada por la cantidad de neuronas ocultas genera una cantidad de pesos enorme para una sola capa. Esto satura la memoria de la GPU y vuelve el entrenamiento ineficiente.

2. Pérdida de la estructura espacial: La MLP requiere aplanar la imagen en un vector unidimensional, lo que destruye la relación de cercanía entre los píxeles (la red olvida qué píxel estaba arriba, abajo o al lado de otro).

En este escenario, una CNN se vuelve muchísimo más eficiente gracias al weight sharing (compartición de pesos). Al deslizar pequeños filtros (kernels) sobre la imagen, la CNN reutiliza los mismos parámetros para detectar características (como bordes o texturas) en cualquier parte de la imagen, reduciendo drásticamente el número de pesos y preservando la estructura espacial bidimensional del dato.

- ¿Qué hace la capa `Flatten()` al principio de la red?

Como se mencionó en las preguntas anteriores, en la red MLP se necesita una entrada unidimensional, por lo que para una entrada como una imagen que es posible verla como una matriz 2D (3D si tenemos en cuenta RGB o el formato que sea) se debe convertir en unidimensional. Para que los datos conserven la relación espacial de la imagen en el formato unidimensional, se debe aplicar la función `Flatten()`.

- ¿Qué función de activación se usó? ¿Por qué no usamos `Sigmoid` o `Tanh`?

Se utiliza la función de activación ReLU, debido a que su derivada es constante e igual a uno para todos los valores positivos, lo cual radica eficazmente el problema del desvanecimiento del gradiente. 

No se utilizan funciones como `Sigmoid` o `Tanh` en las capas ocultas porque sus curvas se saturan en los extremos, lo que significa que para valores de entrada muy altos o muy bajos sus derivadas se vuelven prácticamente cero. Esto provoca que durante la retropropagación, los gradientes disminuyan exponencialmente a través de las capas, deteniendo el aprendizaje de la red.

- ¿Qué parámetro del modelo deberíamos cambiar si aumentamos el tamaño de entrada de la imagen?

Para cambiar el tamaño de entrada de la red, se debe modificar el input_size, que corresponde al primer parámetro de la función `nn.Linear(input_size, 512)`.



## 3. Entrenamiento y Optimización
- ¿Qué hace `optimizer.zero_grad()`?

Lo que hace esta función es limpiar la memoria de los gradientes calculados. Esto permite evitar acumulación de errores en los pesos. Además, en cada iteración del loop, antes de calcular la pérdida y de hacer la retropropagación (`loss.backward()`), se debe utilizar para asegurar de que el lote actual empiece desde cero.


- ¿Por qué usamos `CrossEntropyLoss()` en este caso?

Usamos `CrossEntropyLoss()` porque es la función de pérdida ideal para problemas de clasificación multiclase, donde cada imagen pertenece a una sola clase. Su gran ventaja es que hace dos tareas clave en un solo paso: 

1. Toma las salidas crudas de la red y las transforma en probabilidades.

2. Calcula el error comparando esas predicciones con la etiqueta real, castigando con mucha fuerza a la red si está segura de una respuesta incorrecta.

- ¿Cómo afecta la elección del tamaño de batch (`batch_size`) al entrenamiento?

El `batch_size` es el número de imágenes que la red neuronal procesa de forma simultánea antes de calcular el error y actualizar sus pesos a través del optimizador. Si se escoge un valor chico, el tiempo computacional será más pequeño, pero los resultados no serán ruidosos. Mientras que a mayor `batch_size`, los resultados mejor, ya que hay 'más resolución', pero se paga con tiempo de cómputo. 

- ¿Qué pasaría si no usamos `model.eval()` durante la validación?

`model.eval()` setea el modelo en modo validación. Si este falta, el problema es que se estaría buscando validar, pero el modelo seguiría en modo entrenamiento. Estos dos 'estados/modos' son dos formas distintas de aproximar los cálculos. En entrenamiento se simulan dificultades para lograr ir mejorando el modelo, mientras que en validación se busca testear el rendimiento actual del modelo.

## 4. Validación y Evaluación
- ¿Qué significa una accuracy del 70% en validación pero 90% en entrenamiento?

Este es un comportamiento correspondiente al overfitting. El modelo se adapta demasiado a la data del entrenamiento y esto conduce a errores en la validación.

- ¿Qué otras métricas podrían ser más relevantes que accuracy en un problema real?

En un problema real, especialmente si hay un gran desbalance de datos o implicancias médicas, las métricas más relevantes son el Recall (Sensibilidad), la Precision (Precisión) y el F1-Score. El accuracy solo mide cuantas predicciones correctas realizó de la cantidad de imágenes totales, lo que no garantiza que sea una seguridad, ya que puede haber desbalance de clases o repetición de data.

- ¿Qué información útil nos da una matriz de confusión que no nos da la accuracy?

Mientras que la Accuracy te da una única nota general sin explicar en qué se equivocó el modelo, la matriz de confusión te abre el mapa completo de los errores. Éste te permite realizar un análisis de los errores y comprender donde esta fallando. Es capaz de mostrar el desglose fila por fila de las confusiones específicas del modelo: Marca exactamente qué clase se está tomando como otra.

- En el reporte de clasificación, ¿qué representan `precision`, `recall` y `f1-score`?

El Recall mide cuántos casos positivos reales logró detectar el modelo. En nuestro caso, busca evitar dejar enfermedades sin diagnosticar. La Precision mide qué tan confiable es el modelo cuando asegura haber hecho una predicción. Se deben evitar falsas alarmas que saturen el sistema. El F1-Score actúa como el balance perfecto entre ambas, otorgando una nota final honesta sobre el rendimiento en cada enfermedad, sin dejarse engañar por las clases que tienen miles de imágenes más que el resto.


## 5. TensorBoard y Logging
- ¿Qué ventajas tiene usar TensorBoard durante el entrenamiento?

Tensorboard permite monitorear el comportamiento del modelo en tiempo real mediante gráficas interactivas. En lugar de mirar la consola y realizar muchos prints, TensorBoard propone una interfaz gráfica que muestra como sube la precisión (Accuracy) y cómo baja el error (Loss) época por época. Esto te permite detectar si el modelo dejó de aprender, si está sufriendo de overfitting, o si hay un error en el código.

- ¿Qué diferencias hay entre loguear `add_scalar`, `add_image` y `add_text`?

Cada una de estas funciones sirve para registrar un tipo de dato diferente en el panel de Tensorboard: `add_scalar` se usa para valores numéricos únicos que cambian en el tiempo, como el número de la pérdida o la precisión de cada lote. `add_image` sirve para enviar matrices de píxeles (imágenes) al panel, permitiéndo visualizar qué está viendo la red. `add_text` permite guardar notas en formato de texto, como la arquitectura de la red, los hiperparámetros o el reporte de clasificación de scikit-learn, dejando un registro escrito de la configuración de ese run.

- ¿Por qué es útil guardar visualmente las imágenes de validación en TensorBoard?

Esto es útil ya que te permite hacer un control sobre las predicciones de la red. Al guardar las fotos de validación y las etiquetas, junto con la predicción del modelo, se puede entender que tipo de imágenes está interpretando correctamente el modelo, y cuales no.

- ¿Cómo se puede comparar el desempeño de distintos experimentos en TensorBoard?

Para comparar distintos experimentos, TensorBoard utiliza un sistema de filtros por carpetas y curvas de colores en una misma gráfica. Si se guarda cada test en una subcarpeta diferente (por ejemplo, runs/modelo_base y runs/modelo_with_batchnorm), TensorBoard las lee en simultáneo y dibuja las curvas de ambos entrenamientos superpuestas en la misma gráfica utilizando colores distintos. Esto permite comparar de forma directa cuál de las configuraciones logró bajar el error más rápido o alcanzar mayor precisión.

## 6. Generalización y Transferencia
- ¿Qué cambios habría que hacer si quisiéramos aplicar este mismo modelo a un dataset con 100 clases?

Para escalar el modelo a 100 clases, el único cambio obligatorio en la arquitectura de la red es modificar el número de neuronas de salida de la última capa lineal (nn.Linear) para que sea exactamente 100. Además, se debe asegurar de que la función de pérdida y el reporte de clasificación mapeen correctamente las 100 nuevas etiquetas. También, es muy probable que se necesite aumentar la capacidad de la red, aumentando la cantidad de capas ocultas, y el tiempo de entrenamiento, ya que clasificar entre 100 categorías tiene mayor complejidad que hacerlo entre 4 o 5.

- ¿Por qué una CNN suele ser más adecuada que una MLP para clasificación de imágenes?

Una CNN posee invarianza espacial y comparte pesos a través de sus convoluciones, lo que le permite detectar patrones visuales locales sin imporar en zona de la imagen aparezcan. Por el contrario, una MLP requiere aplanar la imagen en un vector unidimensional muy largo, lo cual destruye la relación de vecindad entre los píxeles (fenómeno que se relaciona en la bidimensionalidad de la matriz).

Además, usar una MLP con imágenes grandes genera que la cantidad de conexiones y parámetros crezca exponencialmente, volviendo al modelo imposible de entrenar, mientras que la CNN se mantiene compacta y eficiente.

- ¿Qué problema podríamos tener si entrenamos este modelo con muy pocas imágenes por clase?

Lo más probable que ocurra si se entrena con muy pocas clases es que ocurra overfitting, ya que el modelo aprendió con pocos ejemplos y es más difícil generar un balance cuando hay pocas imágenes. En cambio, con muchas imágenes, es más fácil lograr un balance, gracias a que los datos serán más homogéneos y cercanos a la realidad (sin la necesidad de ajustar a mano).

- ¿Cómo podríamos adaptar este pipeline para imágenes en escala de grises?

Para adaptar el pipeline a escala de grises, el cambio principal es modificar los canales de entrada de 3 (RGB) a 1 (Escala de grises), ya que un solo canal es suficiente para representar toda la información de la imagen, donde cada píxel toma un valor de intensidad que va desde el negro absoluto hasta el blanco, pasando por todos los grises intermedios. A nivel de código, este cambio se debe aplicar tanto en las transformaciones de carga de datos (transforms) para que conviertan las imágenes a un solo canal, como en la primera capa de la red neuronal, configurando sus parámetros de entrada para que procesen una dimensión de tamaño 1 en lugar de 3.

## 7. Regularización

### Preguntas teóricas:
- ¿Qué es la regularización en el contexto del entrenamiento de redes neuronales?

La regularización se basa en un conjunto de métodos para lograr los ajustes correctos a la hora de entrenar una red neuronal. Ésta se utiliza para corregir el overfitting, y en lugar de conseguir más datos, se utilizan estos métodos que corrigen el problema del overfitting.
- ¿Cuál es la diferencia entre `Dropout` y regularización `L2` (weight decay)?

Estas técnicas son muy distintas. El dropout se cracteriza por desactivar neuronas de forma alatoria en cada entrenamiento. De esta forma, se consigue que las neuronas sean más independientes entre sí. En cambio, la regularización L2 añade la suma del cuadrado de los pesos multiplicada por un parámetro $\lambda$ a la función de pérdida (loss function). El parámetro $\lambda$ permite controlar la penalización. A partir de este agregado, se penalizan los valores extremos, y los pesos de las neuronas se distribuyen de forma más uniforme, y ninguno termina siendo muy grande comparado a los otros. 

- ¿Qué es `BatchNorm` y cómo ayuda a estabilizar el entrenamiento?

Cuando nuestra red tiene muchas capas, la pequeña varianza en los parámetros de entrada puede suponer una alta varianza en capas finales. Este problema se conoce como internal covariance shift. Esto se puede entender de forma que la salida de cada capa es la entrada de la siguiente, por lo que si hay cambios en la entrada de las capas que provoquen cambios en las salidas de estas, las capas tienen que aprender a partir de lo que se conoce como un obejitvo móvil, ya que no estabilidad en el entrenamiento.

BatchNorm propone una capa intermedia para las redes neuronales profundas, de forma que logre estabilizar el entrenamiento al mitigar el internal covariance shift entre las capas de la red. Para lograrlo, BatchNorm calcula la media y la varianza de cada característica dentro de un mini-batch para forzar sus activaciones a una escala estándar (media 0 y varianza 1), aplicando luego una transformación con dos parámetros entrenables (γ y β) que preservan la capacidad no lineal del modelo. Este proceso evita que los gradientes se desvanezcan o exploten, permitiendo el uso de tasas de aprendizaje más altas, reduciendo la dependencia de una inicialización perfecta de los pesos y actuando además como un regularizador ligero que acelera drásticamente la convergencia del entrenamiento.

- ¿Cómo se relaciona `BatchNorm` con la velocidad de convergencia?
- ¿Puede `BatchNorm` actuar como regularizador? ¿Por qué?
- ¿Qué efectos visuales podrías observar en TensorBoard si hay overfitting?
- ¿Cómo ayuda la regularización a mejorar la generalización del modelo?

### Actividades de modificación:
1. Agregar Dropout en la arquitectura MLP:
   - Insertar capas `nn.Dropout(p=0.5)` entre las capas lineales y activaciones.
   - Comparar los resultados con y sin `Dropout`.

2. Agregar Batch Normalization:
   - Insertar `nn.BatchNorm1d(...)` después de cada capa `Linear` y antes de la activación:
     ```python
     self.net = nn.Sequential(
         nn.Flatten(),
         nn.Linear(in_features, 512),
         nn.BatchNorm1d(512),
         nn.ReLU(),
         nn.Dropout(0.5),
         nn.Linear(512, 256),
         nn.BatchNorm1d(256),
         nn.ReLU(),
         nn.Dropout(0.5),
         nn.Linear(256, num_classes)
     )
     ```

3. Aplicar Weight Decay (L2):
   - Modificar el optimizador:
     ```python
     optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
     ```

4. Reducir overfitting con data augmentation:
   - Agregar transformaciones en Albumentations como `HorizontalFlip`, `BrightnessContrast`, `ShiftScaleRotate`.

5. Early Stopping (opcional):
   - Implementar un criterio para detener el entrenamiento si la validación no mejora después de N épocas.

### Preguntas prácticas:
- ¿Qué efecto tuvo `BatchNorm` en la estabilidad y velocidad del entrenamiento?

Redujo la pérdida de forma drástica desde la primera época, estabilizó los gradientes y aceleró la convergencia del modelo en comparación con el caso sin normalizar.

- ¿Cambió la performance de validación al combinar `BatchNorm` con `Dropout`?

Sí, la precisión aumentó notablemente. BatchNorm aportó la estabilidad numérica inicial necesaria y Dropout controló el sobreajuste, resultando en curvas de validación más altas y constantes.

- ¿Qué combinación de regularizadores dio mejores resultados en tus pruebas?

La combinación de los tres mecanismos juntos: BatchNorm + Dropout + L2 (Weight Decay), ya que obtuvo la menor pérdida y la mayor precisión en el conjunto de validación.

- ¿Notaste cambios en la loss de entrenamiento al usar `BatchNorm`?

Sí, la pérdida disminuyó de manera rápida y continua en las primeras épocas, evitando el estancamiento severo que presentaba el modelo cuando no utilizaba este componente.

## 8. Inicialización de Parámetros

### Preguntas teóricas:
- ¿Por qué es importante la inicialización de los pesos en una red neuronal?

La inicialización de pesos es crucial porque establece el punto de partida del algoritmo de optimización, influyendo directamente en la convergencia del modelo y en la estabilidad del flujo de gradientes durante el entrenamiento. Configurar correctamente estos valores iniciales evita que las activaciones de las neuronas se saturen o se desvanezcan a medida que la información se propaga por la arquitectura, garantizando que la red pueda aprender de manera eficiente y eficiente desde las primeras épocas.

- ¿Qué podría ocurrir si todos los pesos se inicializan con el mismo valor?

Si todos los pesos se inicializan con el mismo valor, por ejemplo en cero o en una constante, la red neuronal sufre del problema de simetría, donde todas las neuronas de una misma capa oculta calculan exactamente la misma salida y reciben el mismo gradiente durante la retropropagación. Como resultado, todas las neuronas se actualizan de forma idéntica, anulando la capacidad del modelo para aprender características diferentes y reduciendo la red, sin importar su complejidad, al comportamiento equivalente de un modelo lineal de una sola neurona.

- ¿Cuál es la diferencia entre las inicializaciones de Xavier (Glorot) y He?

La diferencia fundamental radica en cómo escalan la varianza de los pesos iniciales en función del número de neuronas de entrada ($n_{in}$) y salida ($n_{out}$) de cada capa. Mientras que la inicialización de Xavier está diseñada para mantener la varianza constante en activaciones lineales o simétricas como la tangente hiperbólica aplicando la fórmula $\text{Var}(W) = \frac{2}{n_{in} + n_{out}}$, la inicialización de He duplica esa varianza utilizando $\text{Var}(W) = \frac{2}{n_{in}}$ para compensar la pérdida de información que ocurre cuando las funciones de activación no lineales descartan la mitad de las entradas.


- ¿Por qué en una red con ReLU suele usarse la inicialización de He?

En redes que implementan la función de activación ReLU, se utiliza la inicialización de He porque esta función anula todos los valores de entrada negativos, reduciendo la varianza de la señal a la mitad en cada capa. Al duplicar la varianza de los pesos iniciales en comparación con Xavier, el método de He compensa matemáticamente este "apagado" de neuronas, logrando que la escala de las activaciones y de los gradientes se mantenga estable a lo largo de las capas profundas y previniendo el desvanecimiento del gradiente.


- ¿Qué capas de una red requieren inicialización explícita y cuáles no?

Las capas que requieren inicialización explícita son aquellas que poseen parámetros entrenables encargados de transformar linealmente los datos, tales como las capas densas (totalmente conectadas), las capas convolucionales y las capas recurrentes. Por el contrario, las capas que realizan funciones puramente matemáticas, estructurales o estadísticas sin almacenar pesos propios —como las capas de activación (ReLU, Sigmoid), las de reducción (MaxPooling), las de regularización (Dropout) y las de aplanado (Flatten)— no requieren ningún tipo de inicialización.








### Actividades de modificación:
1. Agregar inicialización manual en el modelo:
   - En la clase `MLP`, agregar un método `init_weights` que inicialice cada capa:
     ```python
     def init_weights(self):
         for m in self.modules():
             if isinstance(m, nn.Linear):
                 nn.init.kaiming_normal_(m.weight)
                 nn.init.zeros_(m.bias)
     ```

2. Probar distintas estrategias de inicialización:
   - Xavier (`nn.init.xavier_uniform_`)
   - He (`nn.init.kaiming_normal_`)
   - Aleatoria uniforme (`nn.init.uniform_`)
   - Comparar la estabilidad y velocidad del entrenamiento.

3. Visualizar pesos en TensorBoard:
   - Agregar esta línea en la primera época para observar los histogramas:
     ```python
     for name, param in model.named_parameters():
         writer.add_histogram(name, param, epoch)
     ```

### Preguntas prácticas:
- ¿Qué diferencias notaste en la convergencia del modelo según la inicialización?

La inicialización Uniform mostró un descenso más rápido en el entrenamiento, pero Xavier logró una convergencia más equilibrada a largo plazo en las métricas de validación.

- ¿Alguna inicialización provocó inestabilidad (pérdida muy alta o NaNs)?

Con todos los regularizadores activos no se registraron NaNs, pero al inicio del proceso, las diferencias en la escala de los pesos generaban desfases en los valores iniciales de la pérdida.

- ¿Qué impacto tiene la inicialización sobre las métricas de validación?

Define el límite de rendimiento general del modelo. En esta arquitectura, Xavier obtuvo la precisión de validación más alta, demostrando una mejor capacidad de generalización.

- ¿Por qué `bias` se suele inicializar en cero?

Porque al comienzo del entrenamiento no hay una justificación matemática para inclinar la salida de las neuronas hacia ninguna dirección. Establecerlo en cero garantiza que el aprendizaje dependa inicialmente de los pesos (weights).
