# Barabanov norms

Here are collected some scripts in the MATLAB and Python languages for constructing Barabanov norms of sets of 2x2 matrices and computing the associated characteristics:

- joint/generalized spectral radius of a set of matrices,

- extreme trajectories -- trajectories on which the maximum growth rate of matrix products in the Barabanov norm is reached,

- construction of index sequences of extreme trajectories,

- calculation of the frequency of occurrence of each matrix from the matrix set in the corresponding extreme trajectories.

A modified implementation of the max-relaxation algorithm described in the preprint [arXiv:2112.00391](https://arxiv.org/abs/2112.00391) is used to construct the Barabanov norms. An improvement in the implementation of the algorithm compared to the test case described in [arXiv:2112.00391](https://arxiv.org/abs/2112.00391) is achieved by using tools for working with polygonal objects (the `shapely` package in Python, the `polyshape` object in MATLAB), which leads to the possibility of iteratively constructing polygonal approximations of unit spheres of the Barabanov norms without loss of accuracy.

The main scripts of the proposed set of scripts are

### barnorm-rot.py

This script computes all the above characteristics. In the suggested version of the script, the defaults are made for matrices that are similar to the matrices of plane rotations. One can also use this script for other sets of matrices by simply redefining the matrix elements. In addition to the calculation functions described above, this script implements:

- the visualization of the unit sphere of the constructed Barabanov norm,
- the visualization of an example of an extreme trajectory,
- the visualization of the angular function associated with the process of constructing extreme trajectories.

### barnorm-sturm.py

This script is a slight modification of the script **barnorm-rot.py**, which can be used to compute all of the above characteristics for sets of two matrices, one of which is nonnegative lower-triangular matrix and the other nonnegative upper-triangular matrix. This case is the most theoretically studied and for it has been proved that the index sequences of the extreme trajectories are Sturmian. In addition to the computational functions described above for the **barnorm-rot.py** script, this script implements

- a more convenient additional visualization of the angular function.

### MATLAB-scripts

These are earlier versions of the scripts that are less detailed in terms of visualization and computed characteristics.

### PDF-files

These files contain examples of output files for the scripts **barnorm-rot.py** and **barnorm-sturm.py** for different parameter sets.

---

Здесь собраны некоторые скрипты на языках MATLAB и Python, предназначенные для построения норм Барабанова наборов 2x2 матриц, а также вычисления связанных с ними характеристик:

- совместного/обобщенного спектрального радиуса набора матриц,
- экстремальных траекторий -- траекторий, на которых достигаетмя максимальная скорость роста матричных произведений в норме Барабанова,
- построения индексных последовательностей экстремальных траекторий,
- вычисления частот появления каждой матрицы из матричного множества в соответствующих экстремальных траекториях.

Для построения норм Барабанова применяется модифицированная реализации алгоритма max-релаксации, описанного в препринте [arXiv:2112.00391](https://arxiv.org/abs/2112.00391). Улучшение реализации алгоритма по сравнению с тестовым примером, описанным в [arXiv:2112.00391](https://arxiv.org/abs/2112.00391), достигается за счет применения средств работы с полигональными объектами (пакет ``shapely`` в языке Python, объект ``polyshape`` в MATLAB), что приводит к возможности итерационного построения полигональных приближений единичных сфер норм Барабанова без потери точности.

Основными в предлагаемом наборе скриптов являются

### barnorm-rot.py

Этот скрипт осуществляет вычисление всех характеристик, упомянутых выше. В предлагаемом варианте скрипта предварительные настройки выполнены для матриц, подобных матрицам поворотов плоскости. Применение этого скрипта для анализа других наборов матриц может быть выполнено простым переопределением элементов матриц. Помимо вычислительных функций, описанных выше, данный скрипт осуществляет:

- визуализацию единичной сферы построенной нормы Барабанова,
- визуализацию примера экстремальной траектории,
- визуализацию угловой функции, связанной с процессом построения экстремальных траекторий.

### barnorm-sturm.py

Этот скрипт является незначительным изменением скрипта **barnorm-rot.py**, предназначенным для вычисления всех упомянутых выше характеристик для наборов из двух матриц, одна из которых является неотрицательной нижне-треугольной, а другая неотрицательной верхне-треугольной. Именно этот случай является наиболее теоретически исследованным и для него доказано, что индексные последовательности экстремальных траекторий штурмовы. Помимо вычислительных функций, описанных выше для скрипта **barnorm-rot.py**, данный скрипт осуществляет

- более удобную дополнительную визуализации угловой функции.

### MATLAB-скрипты

Это более ранние версии скриптов, менее детальные в плане визуализации и наборов вычисляемых характеристик.

### PDF-файлы

В данных файлах представлены примеры работы скриптов **barnorm-rot.py** и **barnorm-sturm.py** для разных наборов параметров.
