# geometry-selector

Библиотека для выбора геометрической модели по матрице попарных расстояний.

Проект выполнен в рамках ВКР студента Козлова Дмитрия группы M3534 КТ ИТМО. Библиотека сравнивает евклидову, гиперболическую и сферическую модели вложения, 
автоматически подбирает параметр кривизны для гиперболического и радиус для сферического случаев, а затем выбирает наиболее подходящую геометрию по значению stress.

## Возможности

- выбор геометрии по матрице расстояний;
- поддержка моделей евклидовой, гиперболической и сферической;
- спектральный подбор параметров кривизны;
- опциональная локальная дооптимизация через флаг `do_plus=True`;
- вывод таблицы кандидатов и текстовой рекомендации.

## Установка

Из корня репозитория:

```bash
pip install -e .
```

Для запуска тестов:

```bash
pip install -e ".[dev]"
pytest
```

## Простейший пример

```python
import numpy as np

from geomselect import pairwise_euclidean, select_geometry

rng = np.random.default_rng(0)
X = rng.normal(size=(100, 2))

D = pairwise_euclidean(X)

result = select_geometry(D, d=2)

print(result.selected_geometry)
print(result.recommendation)
print(result.candidate_table)
```

## Пример с дооптимизацией

```python
result = select_geometry(
    D,
    d=2,
    do_plus=True,
    plus_maxiter_hyper=100,
    plus_maxiter_sphere=100,
    rollback_plus=True,
)

print(result.selected_geometry)
print(result.candidate_table)
```

## Что возвращает select_geometry

Функция `select_geometry` возвращает объект `SelectionResult`.

Основные поля:

- `result.selected_geometry` — выбранная геометрия;
- `result.selected` — выбранный кандидат;
- `result.candidates` — все кандидаты;
- `result.candidate_table` — таблица сравнения моделей;
- `result.recommendation` — текстовая рекомендация;
- `result.metadata` — дополнительная диагностическая информация.


## Статус

Библиотека является исследовательским прототипом, созданным для экспериментов в рамках ВКР. 
Основное назначение — демонстрация и проведение вычислительных экспериментов.
