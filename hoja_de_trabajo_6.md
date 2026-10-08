# CC2017 Modelación y Simulación

## Hoja de Trabajo 6

### Instrucciones

* Esta es una actividad en grupos de no más de 3 integrantes.

* Recuerden unirse al grupo de Canvas.

* No se permitirá ni se aceptará cualquier indicio de copia. De presentarse, se procederá según el reglamento correspondiente.

* Tendrán hasta el día indicado en Canvas.

* No se confíen, aprovechen el tiempo en clase para entender todos los ejercicios y avanzar lo más posible.

### Contexto del Problema

Dos plataformas digitales compiten por la misma base de usuarios. Sea $x(t)$ la base de usuarios de la plataforma A y $y(t)$ la de la plataforma B, ambas medidas en unidades de la capacidad de mercado que cada plataforma alcanzaría si estuviera sola. Cada plataforma crece por sí sola de forma logística y es frenada por la rivalidad de la otra:

$$
\dot{x} = r_1 x (1 - x - \alpha_{12} y)
$$

$$
\dot{y} = r_2 y (1 - y - \alpha_{21} x)
$$

donde $r_1, r_2 > 0$ son las tasas de crecimiento, $\alpha_{12}$ mide cuánto frena B a A, y $\alpha_{21}$ mide cuánto frena A a B.

Según los parámetros, el sistema puede terminar en coexistencia, en la exclusión de una plataforma o en un régimen donde gana quien llegue primero. Su trabajo es diagnosticar el régimen de su grupo, predecir su comportamiento antes de simularlo, verificar la predicción con simulación propia y estudiar un sistema distinto que sí tiene un punto de no retorno.

#### Parámetros de su grupo

Sea $g$ el número de su grupo. El conjunto de parámetros se determina así:

| Conjunto | $\alpha_{12}$ | $\alpha_{21}$ | 
| ----- | ----- | ----- | 
| **1** | 0.5 | 0.6 | 
| **2** | 1.4 | 1.5 | 
| **3** | 0.6 | 1.5 | 
| **4** | 1.5 | 0.6 | 

Su conjunto es el número $((g - 1) \bmod 4) + 1$. Además, $r_1 = 1$ y $r_2 = 0.5 + 0.1 ((g - 1) \bmod 5)$. Indique explícitamente su número de grupo, su conjunto y sus valores de $r_2$ al inicio del reporte.

#### Estados iniciales de referencia (se usan en todos los tasks)

| Etiqueta | $(x_0, y_0)$ | 
| ----- | ----- | 
| **P1** | (0.05, 0.05) | 
| **P2** | (0.80, 0.30) | 
| **P3** | (0.30, 0.80) | 
| **P4** | (0.60, 0.60) | 

#### Herramientas

Python con NumPy, SciPy y Matplotlib. SymPy puede usarse únicamente para verificar derivaciones simbólicas que usted haya hecho a mano. Está permitido usar `solve_ivp` de SciPy solo como referencia de comparación donde se indique. Todas las simulaciones deben fijar una semilla y ser reproducibles.

## Task 1 (Entrega Parcial)

### Task 1.1

Para los equilibrios del sistema, realice:
a. Derive a mano, para parámetros generales, los cuatro equilibrios del sistema. Los tres primeros son $(0,0)$, $(1,0)$ y $(0,1)$. Obtenga el cuarto, interior, resolviendo el sistema lineal $x + \alpha_{12} y = 1$, $y + \alpha_{21} x = 1$.
b. Escriba la condición sobre $\alpha_{12}$ y $\alpha_{21}$ bajo la cual el equilibrio interior tiene ambas coordenadas positivas, es decir, es biológicamente posible. Demuestre que, con ambos parámetros distintos de uno, esto ocurre si y solo si ambos son menores que uno o ambos son mayores que uno.
c. Evalúe los cuatro equilibrios con los parámetros de su grupo. Indique cuáles son factibles y cuáles no.

### Task 1.2

Respecto a la clasificación por estabilidad lineal, realice:
a. Calcule la matriz jacobiana del sistema para parámetros generales, derivando cada ecuación respecto a cada variable.
b. Evalúe la jacobiana en cada equilibrio factible de su grupo. Para cada uno reporte la traza $\tau$, el determinante $\Delta$, el discriminante $\tau^2 - 4\Delta$, los valores propios y la clasificación (nodo, silla, foco o centro, estable o inestable). Para los equilibrios sobre los ejes, obtenga los valores propios de forma analítica, aprovechando que la matriz es triangular.
c. Ubique cada equilibrio de su grupo sobre el plano traza y determinante. Explique en un párrafo qué indica la posición de cada uno sobre la sensibilidad de su clasificación a pequeños errores en los parámetros.
d. Con los resultados anteriores, determine el régimen cualitativo de su grupo: coexistencia, biestabilidad, o exclusión de una de las plataformas. Justifique usando únicamente los valores propios, sin simular.

### Task 1.3

Esta sección debe resolverse completa antes de empezar el Task 2.
a. Escriba las ecuaciones de las nulclinas de $x$ y de $y$ para sus parámetros. Dibújelas a mano o con código, con sus intersecciones marcadas, en el cuadrante $x, y \ge 0$.
b. En cada región en que las nulclinas dividen el cuadrante, determine los signos de $\dot{x}$ y $\dot{y}$ y dibuje la dirección de la flecha del campo vectorial. Marque los ejes $x=0$ e $y=0$ e indique si son conjuntos invariantes, argumentando con las ecuaciones.
c. **Predicción:** Para cada uno de los estados iniciales P1, P2, P3 y P4, prediga a cuál equilibrio converge el sistema. Para cada predicción escriba el razonamiento en dos o tres líneas, apoyado en las nulclinas y en los valores propios. Si en alguno de los estados no está seguro, indique que no puede decidir y explique qué información adicional necesitaría. Esta predicción no debe modificarse después de ver cualquier simulación. Las discrepancias se analizan en el Task 2.
d. Prediga el tiempo característico de recuperación del equilibrio estable de su régimen frente a una perturbación pequeña, expresado como $1 / \min \vert{}\mathrm{Re}(\lambda)\vert{}$ sobre sus valores propios estables. Si su régimen tiene dos equilibrios estables, reporte ambos.

### Task 1.4

Un compañero analizó el sistema con $\alpha_{12} = 1.4$, $\alpha_{21} = 1.5$, $r_1 = 1$ y $r_2 = 0.8$, y entregó el siguiente análisis, que se reproduce tal cual lo escribió:

> **Texto del compañero:** El equilibrio interior es $(x^*, y^*) \approx (0.364, 0.455)$. La jacobiana en ese punto es:
> 
>
> $$
> J \approx \begin{pmatrix} -0.364 & -0.509 \\ -0.545 & -0.364 \end{pmatrix}
> $$
>
> 
> Entonces $\tau \approx -0.727$ y $\Delta = J_{11} J_{22} + J_{12} J_{21} \approx 0.41$. Como $\Delta > 0$ y $\tau < 0$, y además $\tau^2 - 4\Delta < 0$, el equilibrio es un foco estable. Por lo tanto, las dos plataformas coexisten oscilando alrededor de $(x^*, y^*)$ y el sistema es robusto a perturbaciones.

a. Encuentre el error en el análisis y explique por qué es un error, no una cuestión de criterio.
b. Corrija el cálculo. Reporte el valor correcto de $\Delta$, la clasificación correcta del equilibrio y la conclusión correcta sobre el régimen.
c. El compañero tiene además un error de interpretación que no depende del cálculo anterior: incluso si el equilibrio interior fuera un foco estable con esos parámetros, la afirmación final contiene una conclusión que no se sigue. Identifíquela y explique qué debió escribir en su lugar.

## Task 2 (Entrega Parcial)

### Task 2.1

Realice:
a. Implemente desde cero, sin usar `solve_ivp` ni ninguna función de integración de librería, el método de Euler y el método de Runge-Kutta de cuarto orden para el sistema de su grupo. Cada integrador debe recibir el campo $f$, el estado inicial, el paso $h$ y el tiempo final.
b. Use `solve_ivp` con tolerancias de $10^{-12}$ (método DOP853) como referencia para el estado P2 con tiempo final $T = 10$. Calcule el error de su Euler y de su Runge-Kutta para los pasos $h \in \{0.2, 0.1, 0.05, 0.025, 0.0125\}$.
c. Grafique el error contra el paso en escala log-log para ambos métodos. Estime la pendiente de cada recta por regresión y compárela con el orden teórico de cada método. Explique cualquier desviación.

### Task 2.2

Realice:
a. Construya un retrato de fases del cuadrante $[0, 1.5] \times [0, 1.5]$ con el campo vectorial (flechas normalizadas), las dos nulclinas y al menos doce trayectorias calculadas con su Runge-Kutta desde estados iniciales distribuidos de forma que muestren la estructura completa, incluyendo estados muy cerca de los ejes.
b. Marque los equilibrios con color según su tipo y marque los cuatro estados de referencia P1 a P4 con sus trayectorias resaltadas.
c. Compare cada una de sus cuatro predicciones del Task 1.3 con el resultado de la simulación. Para cada discrepancia, no corrija la predicción: explique qué parte del razonamiento falló. Si no hubo discrepancias, escoja la predicción en la que estuvo menos seguro y explique por qué acertó.

### Task 2.3

Conteste:
a. Construya una malla de $60 \times 60$ estados iniciales uniformes en $[0.01, 1.5]^2$ y simule cada uno hasta que se estabilice en un equilibrio. Defina un criterio de convergencia y justifíquelo. Coloree la malla según el equilibrio al que llega cada estado inicial y grafique el mapa de cuencas.
b. Reporte la fracción del cuadrante que pertenece a la cuenca de cada equilibrio estable de su régimen.
c. Si su régimen tiene dos equilibrios estables, estime numéricamente la frontera entre ambas cuencas por bisección a lo largo de al menos cinco segmentos distintos y grafíquela. Compare su forma con la variedad estable del equilibrio interior, calculando la dirección del vector propio asociado al valor propio negativo y verificando si la frontera numérica es tangente a esa dirección cerca de la silla.
d. Si su régimen no tiene dos equilibrios estables, explique qué muestra su mapa de cuencas y por qué tiene esa forma, y qué papel cumplen los ejes en ese resultado.

### Task 2.4

Conteste:
a. Perturbe el equilibrio estable de su régimen por una distancia de $10^{-4}$ en ocho direcciones distintas, igualmente espaciadas, que mantengan el estado dentro del cuadrante $x, y \ge 0$ (si el equilibrio está sobre un eje, solo hay direcciones disponibles en un cuarto de plano), y simule con su Runge-Kutta. Si su régimen tiene dos equilibrios estables, repita el análisis en ambos. En cada caso ajuste una recta a $\ln \Vert{}x(t) - x^*\Vert{}$ en la ventana en la que la norma cae de $10^{-4}$ a $10^{-6}$ y obtenga la tasa de decaimiento.
b. Compare las tasas medidas con $\min \vert{}\mathrm{Re}(\lambda)\vert{}$ calculado en el Task 1.2. Reporte el error relativo. Explique por qué algunas direcciones iniciales dan una tasa distinta durante el transitorio inicial pero coinciden con la teoría a tiempos largos.
c. Para su equilibrio estable, derive a mano la condición sobre $h$ para que Euler converja, desarrollando $\vert{}1 + h\lambda\vert{} < 1$ en términos de la parte real y la norma de $\lambda$. Calcule $h_{\max}$ para cada valor propio y quédese con el más restrictivo.
d. Determine numéricamente el mayor paso con el que su Euler todavía converge al equilibrio desde un estado cercano, por bisección. Compare con el valor teórico del inciso c. Si no coinciden, explique por qué la linealización no es suficiente para predecir el comportamiento numérico lejos del equilibrio.

### Task 2.5

La dirección de una de las plataformas debe elegir entre dos intervenciones: una campaña puntual de adquisición que agrega de golpe una cantidad $\Lambda$ de usuarios a su plataforma, o una acción de diferenciación del producto que reduce la rivalidad que esa plataforma sufre de la otra. Tome como plataforma de interés la que termina con menos usuarios partiendo de P2 en su régimen (en caso de coexistencia, la que tiene menor base en el equilibrio interior).
a. Determine si existe un valor mínimo $\Lambda > 0$ de la campaña que cambia el destino del sistema partiendo de P2. Si existe, estímelo por bisección sobre su simulación y verifíquelo con el mapa de cuencas. Si no existe, demuestre por qué usando la estructura de los equilibrios y de los ejes.
b. Encuentre el valor crítico del parámetro de rivalidad correspondiente (el que frena a esa plataforma) en el que el régimen cambia cualitativamente. Derívelo analíticamente a partir de los valores propios, y compruebe con dos simulaciones, una a cada lado del valor crítico.
c. Recomiende una de las dos intervenciones y argumente cuál es más robusta ante un error en la estimación de los parámetros. Su argumento debe usar el concepto de tiempo de recuperación y la cercanía de los valores propios al eje imaginario, no solo la comparación de efectos.

## Task 3 (Entrega Final)

En este task cambiamos de sistema para estudiar un fenómeno que el modelo de competencia no tiene: un **punto de no retorno**. Se usa un modelo de un lago con reciclaje interno de nutrientes, en el que $x(t)$ es la concentración de nutrientes en el agua (en unidades normalizadas) y $a$ es la carga externa de nutrientes:

$$
\dot{x} = a - b x + \frac{x^2}{x^2 + 1} = f(x; a)
$$

con $r = 1$, $h = 1$ y exponente $p = 2$ ya sustituidos. La tasa de pérdida $b$ depende de su grupo: $b = 0.52 + 0.03 ((g - 1) \bmod 3)$.

### Task 3.1

Realice y conteste:
a. Despeje la carga $a$ en función de $x$ en equilibrio, $a = g(x) = b x - \frac{x^2}{x^2 + 1}$. Demuestre que la estabilidad del equilibrio $x^*$ se decide por el signo de $g'(x^*)$: estable si $g'(x^*) > 0$, inestable si $g'(x^*) < 0$.
b. Calcule numéricamente, para su valor de $b$, los dos puntos donde $g'(x) = 0$ y los valores de carga $a_{\mathrm{col}}$ y $a_{\mathrm{rec}}$ asociados a cada uno, identificando cuál es el colapso y cuál es la recuperación. Reporte el ancho de la región biestable $a_{\mathrm{col}} - a_{\mathrm{rec}}$.
c. Grafique el diagrama de equilibrios ($x^*$ en el eje vertical, $a$ en el horizontal), coloreando de forma distinta las ramas estables y la rama inestable.
d. Verifique el diagrama con simulaciones deterministas: haga subir $a$ muy lentamente desde un valor bajo hasta pasado el colapso, y luego bájelo lentamente hasta pasada la recuperación. Registre en qué valor de $a$ salta el sistema en cada sentido y compárelo con $a_{\mathrm{col}}$ y $a_{\mathrm{rec}}$. Explique cualquier diferencia en términos de la velocidad de cambio de $a$.

### Task 3.2

Realice y conteste:
a. Derive a mano, a partir de la linealización de $f$ alrededor de un equilibrio estable $x^*(a)$, que las fluctuaciones pequeñas inducidas por ruido blanco de intensidad $\sigma$ siguen un proceso de Ornstein-Uhlenbeck $dy = -\kappa y \, dt + \sigma \, dW$ con $\kappa = g'(x^*)$. Obtenga que la varianza estacionaria es $\sigma^2 / (2\kappa)$ y que la autocorrelación a un retardo $\Delta t$ es $e^{-\kappa \Delta t}$. Indique qué ocurre con ambas cantidades cuando $a \rightarrow a_{\mathrm{col}}$.
b. Implemente Euler-Maruyama para $dx = f(x; a(t)) dt + \sigma dW$ con $\sigma = 0.01$, $dt = 0.05$ y una carga que sube lentamente $a(t) = a_0 + v t$, donde $a_0 = 0.5 a_{\mathrm{col}}$ y $v$ se escoge para que $a$ alcance $a_{\mathrm{col}}$ en $t = 2000$. Use como condición inicial el equilibrio de aguas claras para $a_0$. Simule hasta que el sistema salte a la rama de alta concentración.
c. Sobre la parte de la serie anterior al salto, elimine la tendencia (justifique el método que use) y calcule, con una ventana móvil cuyo tamaño usted justificará, la varianza y la autocorrelación de retardo uno. Grafique ambas contra el tiempo, con el valor teórico del inciso a superpuesto.
d. Repita para 100 realizaciones independientes. Para cada una calcule el tau de Kendall entre cada indicador y el tiempo. Reporte la fracción de realizaciones en las que la tendencia es positiva y significativa al nivel de 5%.
e. Como control, repita el inciso d con una carga constante $a = 0.5 a_{\mathrm{col}}$, durante el mismo tiempo. Reporte la fracción de realizaciones que dan una señal de alerta positiva significativa. Esto estima la tasa de falsas alarmas. Interprete: ¿qué tan confiable es el indicador como sistema de alarma, considerando ambas fracciones?
f. Reporte cómo cambian los resultados al duplicar $\sigma$ y al reducir a la mitad el tiempo de subida de $a$. Explique cada cambio usando la teoría del inciso a y el supuesto de variación lenta del parámetro.

### Task 3.3

Busque en Google Scholar o Scopus un paper publicado entre 2021 y 2026 que use señales de alerta temprana, análisis de estabilidad o análisis de bifurcaciones en un sistema real, por ejemplo un ecosistema, el clima, un sistema financiero, un sistema de salud o una red eléctrica.
a. Cite el paper en formato APA. Identifique el sistema, la variable que se monitorea, los indicadores que se calculan y el tipo de transición que se busca detectar.
b. Evalúe si los supuestos del paper coinciden con los supuestos de su Task 3.2: variación lenta del parámetro, ruido estacionario, ventana de cálculo y método de detrendización. Señale al menos dos supuestos que el paper no verifica o no puede verificar con sus datos, y qué efecto tendría su incumplimiento sobre las falsas alarmas.
c. Compare la tasa de detección y la de falsas alarmas del paper, si las reporta, con las que usted obtuvo en el Task 3.2. Si el paper no las reporta, explique qué implica eso para la credibilidad de su conclusión.
d. Proponga una modificación concreta a la metodología del paper, que se pueda implementar con los datos que ya tiene, para reducir falsas alarmas. Justifíquela con lo aprendido en el Task 3.2.

### Task 3.4

Escriba un párrafo de no más de 150 palabras para el administrador de un lago, que no tiene formación en modelación, en el que comunique:

* Qué le dicen sus indicadores si empiezan a subir.

* Qué no le dicen: por qué una señal de alerta no garantiza un colapso y por qué su ausencia no garantiza seguridad.

* Qué decisión concreta tomaría usted si observara la señal, considerando que recuperar el lago cuesta más que evitar su colapso.

*Nota: No use notación matemática en este párrafo.*

## Entregas en Canvas

1. Documento PDF con las respuestas a cada task.

2. En la entrega parcial se espera que entreguen lo señalado. En la entrega final deben entregar **TODOS LOS TASKS**.

3. Archivo `.ipynb`, o link a repositorio de GitHub (No se aceptan entregas por otros medios).
   a. El código debe estar comentado explicando la relación con las fórmulas de las diapositivas.

## Evaluación

| Criterio | Punteo | 
| ----- | ----- | 
| **Task 1** | 0.90 pt | 
| **Task 2** | 1.10 pt | 
| **Task 3** | 1.00 pt | 
| **Total** | **3.00 pts** | 
