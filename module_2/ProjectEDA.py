#!/usr/bin/env python
# coding: utf-8

# Проект 2. Разведывательный анализ данных. Итоговое задание
# 
# 1 school — аббревиатура школы, в которой учится ученик
# 2 sex — пол ученика ('F' - женский, 'M' - мужской)
# 3 age — возраст ученика (от 15 до 22)
# 4 address — тип адреса ученика ('U' - городской, 'R' - за городом)
# 5 famsize — размер семьи('LE3' <= 3, 'GT3' >3)
# 6 Pstatus — статус совместного жилья родителей ('T' - живут вместе 'A' - раздельно)
# 7 Medu — образование матери (0 - нет, 1 - 4 класса, 2 - 5-9 классы, 3 - среднее специальное или 11 классов, 4 - высшее)
# 8 Fedu — образование отца (0 - нет, 1 - 4 класса, 2 - 5-9 классы, 3 - среднее специальное или 11 классов, 4 - высшее)
# 9 Mjob — работа матери ('teacher' - учитель, 'health' - сфера здравоохранения, 'services' - гос служба, 'at_home' - не работает, 'other' - другое)
# 10 Fjob — работа отца ('teacher' - учитель, 'health' - сфера здравоохранения, 'services' - гос служба, 'at_home' - не работает, 'other' - другое)
# 11 reason — причина выбора школы ('home' - близость к дому, 'reputation' - репутация школы, 'course' - образовательная программа, 'other' - другое)
# 12 guardian — опекун ('mother' - мать, 'father' - отец, 'other' - другое)
# 13 traveltime — время в пути до школы (1 - <15 мин., 2 - 15-30 мин., 3 - 30-60 мин., 4 - >60 мин.)
# 14 studytime — время на учёбу помимо школы в неделю (1 - <2 часов, 2 - 2-5 часов, 3 - 5-10 часов, 4 - >10 часов)
# 15 failures — количество внеучебных неудач (n, если 1<=n<3, иначе 0)
# 16 schoolsup — дополнительная образовательная поддержка (yes или no)
# 17 famsup — семейная образовательная поддержка (yes или no)
# 18 paid — дополнительные платные занятия по математике (yes или no)
# 19 activities — дополнительные внеучебные занятия (yes или no)
# 20 nursery — посещал детский сад (yes или no)
# 21 higher — хочет получить высшее образование (yes или no)
# 22 internet — наличие интернета дома (yes или no)
# 23 romantic — в романтических отношениях (yes или no)
# 24 famrel — семейные отношения (от 1 - очень плохо до 5 - очень хорошо)
# 25 freetime — свободное время после школы (от 1 - очень мало до 5 - очень мого)
# 26 goout — проведение времени с друзьями (от 1 - очень мало до 5 - очень много)
# 27 health — текущее состояние здоровья (от 1 - очень плохо до 5 - очень хорошо)
# 28 absences — количество пропущенных занятий
# 29 score — баллы по госэкзамену по математике

# Цель: отследить влияние условий жизни учащихся в возрасте от 15 до 22 лет на их успеваемость по математике, чтобы на ранней стадии выявлять студентов, находящихся в группе риска
# Задачи:
# 1.	Первичная обработка данных
# 2.	Устранить выбросы в числовых переменных
# 3.	Оценить количество уникальных значений для номинативных переменных
# 4.	Преобразовать данные при необходимости
# 5.	Провести корреляционный анализ количественных переменных
# 6.	Отобрать не коррелирующие переменные
# 7.	Проанализировать номинативные переменные и устранить те, которые не влияют на предсказываемую величину
# 8.	Сформулировать выводы относительно качества даннызи тех переменных, которые будут использоваться в дальнейшем

# In[71]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import combinations
from scipy.stats import ttest_ind

pd.set_option('display.max_rows', 50)  # выведем больше строк
pd.set_option('display.max_columns', 50)  # выведем больше колонок

math = pd.read_csv('stud_math.csv')

display(math.head(10))
math.info()

# В датасете только 3 колонки без пропусков. Одна лишняя колонка.


# In[72]:


# 1. Первичная обработка данных


# In[73]:


# Лишний столбец с непонятной информацией 'studytime, granular'.
# Шкалы интерпритации результатов нет
display(pd.DataFrame(math['studytime, granular'].value_counts()))

# Убираем его из датасета, как непроверенную информацию
del math['studytime, granular']


# In[74]:


# 2.Проверим все столбцы на выбросы

# age
display(pd.DataFrame(math.age.value_counts()))
math.loc[:, ['age']].info()

# Этот столбец числовой и без пропусков.
# Поэтому посмотрим на его распределение:

math.age.hist()
math.age.describe()


# In[75]:


# Medu
display(pd.DataFrame(math.Medu.value_counts()))
math.loc[:, ['Medu']].info()
# Этот столбец числовой, есть пропуски.

# Fedu
display(pd.DataFrame(math.Fedu.value_counts()))
math.loc[:, ['Fedu']].info()
# Этот столбец числовой, есть пропуски. Есть выбросы.

# traveltime
display(pd.DataFrame(math.traveltime.value_counts()))
math.loc[:, ['traveltime']].info()
# Этот столбец числовой, есть пропуски.

# studytime
display(pd.DataFrame(math.studytime.value_counts()))
math.loc[:, ['studytime']].info()
# Этот столбец числовой, есть пропуски.

# failures
display(pd.DataFrame(math.failures.value_counts()))
math.loc[:, ['failures']].info()
# Этот столбец числовой, есть пропуски.

# famrel
display(pd.DataFrame(math.famrel.value_counts()))
math.loc[:, ['famrel']].info()
# Этот столбец числовой, есть пропуски. Есть выбросы.

# freetime
display(pd.DataFrame(math.freetime.value_counts()))
math.loc[:, ['freetime']].info()
# Этот столбец числовой, есть пропуски.

# goout
display(pd.DataFrame(math.goout.value_counts()))
math.loc[:, ['goout']].info()
# Этот столбец числовой, есть пропуски.

# health
display(pd.DataFrame(math.health.value_counts()))
math.loc[:, ['health']].info()
# Этот столбец числовой, есть пропуски.

# absences
display(pd.DataFrame(math.absences.value_counts()))
math.loc[:, ['absences']].info()
# Этот столбец числовой, есть пропуски. Есть выбросы.

# score
display(pd.DataFrame(math.score.value_counts()))
math.loc[:, ['score']].info()
# Этот столбец числовой, есть пропуски. Есть выбросы в виде оценки 0.0

# В столбцах absences, famrel, Fedu убрать выбросы.


# In[76]:


# 2. Устранение выбросов для числовых переменных


# In[77]:


# absences
math.absences.hist()
math.absences.describe()

# Большая часть значений находится в диапозоне до 20 прогулов


# In[78]:


# рассмотрим распределение более подробно, определим границы выбросов.
# ГРаницы выбросов составили: -12, 20. Оставим только тех, у кого 20 и меньше пропусков.

median = math.absences.median()
IQR = math.absences.quantile(0.75) - math.absences.quantile(0.25)
perc25 = math.absences.quantile(0.25)
perc75 = math.absences.quantile(0.75)
print('25-й перцентиль: {},'.format(perc25), '75-й перцентиль: {},'.format(perc75),
      "IQR: {}, ".format(IQR), "Границы выбросов: [{f}, {l}].".format(f=perc25 - 1.5*IQR, l=perc75 + 1.5*IQR))
math.absences.loc[math.absences.between(perc25 - 1.5*IQR, perc75 + 1.5*IQR)].hist(bins=10, range=(0, 20),
                                                                                  label='IQR')
math.absences.loc[math.absences <= 75].hist(alpha=0.5, bins=10, range=(0, 20),
                                            label='Здравый смысл')
plt.legend()


# In[79]:


# Оставим только те значения, которые меньше 20
math = math.loc[math.absences <= 20]


# In[80]:


# age

# ГРаницы выбросов составили: 13, 21. Оставим только тех, у кого 21 и меньше пропусков.

median = math.age.median()
IQR = math.age.quantile(0.75) - math.age.quantile(0.25)
perc25 = math.age.quantile(0.25)
perc75 = math.age.quantile(0.75)
print('25-й перцентиль: {},'.format(perc25), '75-й перцентиль: {},'.format(perc75),
      "IQR: {}, ".format(IQR), "Границы выбросов: [{f}, {l}].".format(f=perc25 - 1.5*IQR, l=perc75 + 1.5*IQR))
math.age.loc[math.age.between(perc25 - 1.5*IQR, perc75 + 1.5*IQR)].hist(bins=8, range=(15, 22),
                                                                        label='IQR')
math.age.loc[math.age <= 75].hist(alpha=0.5, bins=8, range=(15, 22),
                                  label='Здравый смысл')
plt.legend()


# In[81]:


# Оставим только те значения, которые меньше или равны 21 лет.
math = math.loc[math.age <= 21]


# In[82]:


# 3. Оцениваем количество уникальных значений для нооминативных переменных


# In[83]:


# famrel

# Уберем значение равное -1.0. Т.к. используется шкала от 1 до 5
math = math.loc[math.famrel >= 1]


# In[84]:


# Fedu
# Уберем значение больше 4.0 Т.к. используется шкала от 0 до 4
math = math.loc[math.Fedu <= 4]


# In[85]:


# 4. Преобразовываем данные


# In[86]:


# Обработка номинативных данных

# напишем функцию для замены NaN на None для dtype(object)


def fix_nan(x):
    if pd.isnull(x):
        return None
    if x == 'nan':
        return None
    return x


# In[87]:


# убираем пропуски во всех колонках
math.address = math.address.apply(fix_nan).sort_values()
math.famsize = math.famsize.apply(fix_nan).sort_values()
math.Pstatus = math.Pstatus.apply(fix_nan).sort_values()
math.Mjob = math.Mjob.apply(fix_nan).sort_values()
math.Fjob = math.Fjob.apply(fix_nan).sort_values()
math.reason = math.reason.apply(fix_nan).sort_values()
math.guardian = math.guardian.apply(fix_nan).sort_values()
math.famsup = math.famsup.apply(fix_nan).sort_values()
math.paid = math.paid.apply(fix_nan).sort_values()
math.activities = math.activities.apply(fix_nan).sort_values()
math.nursery = math.nursery.apply(fix_nan).sort_values()
math.higher = math.higher.apply(fix_nan).sort_values()
math.internet = math.internet.apply(fix_nan).sort_values()
math.romantic = math.romantic.apply(fix_nan).sort_values()


# In[88]:


# Убираем в датасете строки с пустыми занчениями в колонке score
math2 = math.dropna(subset=['score'])


# In[89]:


# Убираем значения score = 0
math2 = math2.loc[math.score != 0]


# In[91]:


# 5. Проводим корреляционный анализ количественных переменных


# In[92]:


# строим график pairplot
sns.pairplot(math, kind='reg')


# In[93]:


# 6. создаем матрицу корреляций для числовых столбцов

# уберем нечисловые стлбцы
math_number = math2.drop(['school', 'sex', 'address', 'famsize', 'Pstatus', 'Mjob', 'Fjob', 'reason', 'guardian', 'schoolsup', 'famsup', 'paid',
                          'activities', 'nursery', 'higher', 'internet', 'romantic', 'Medu', 'Fedu', 'traveltime', 'studytime', 'famrel', 'freetime', 'goout', 'health'], axis=1)

# строим таблицу корреляционного анализа
math_number.corr()

# вывод: числовые столбцы не коррелируют между собой, оставим все значения для дальнейшего анализа.


# In[94]:


# 7.Анализ номинативных переменных с помощью боксплотов (функция)
def get_boxplot(column):
    fig, ax = plt.subplots(figsize=(14, 4))
    sns.boxplot(x=column, y='score',
                data=math2.loc[math2.loc[:, column].isin(
                    math2.loc[:, column].value_counts().index[:10])],
                ax=ax)
    plt.xticks(rotation=45)
    ax.set_title('Boxplot for ' + column)
    plt.show()


# In[95]:


# выводим боксплоты
for col in ['school', 'sex', 'address', 'famsize', 'Pstatus', 'Mjob', 
            'Fjob', 'reason', 'guardian', 'schoolsup', 'famsup', 'paid', 
            'activities', 'nursery', 'higher', 'internet', 'romantic', 
            'Medu', 'Fedu', 'traveltime', 'studytime', 'failures', 'famrel', 'freetime', 'goout', 'health']:
    get_boxplot(col)


# In[96]:


# 8. с помощью критерия Стьюдента смотрим влияние номинативных переменных на score (функция)
def get_stat_dif(column):
    cols = math2.loc[:, column].value_counts().index[:10]
    combinations_all = list(combinations(cols, 2))
    for comb in combinations_all:
        if ttest_ind(math2.loc[math2.loc[:, column] == comb[0], 'score'],
                     math2.loc[math2.loc[:, column] == comb[1], 'score']).pvalue \
                <= 0.05/len(combinations_all):  # Учли поправку Бонферони
            print('Найдены статистически значимые различия для колонки', column)
            break


# In[97]:


# применение функции
for col in ['school', 'sex', 'address', 'famsize', 
            'Pstatus', 'Mjob', 'Fjob', 'reason', 
            'guardian', 'schoolsup', 'famsup', 'paid', 
            'activities', 'nursery', 'higher', 'internet', 
            'romantic', 'Medu', 'Fedu', 'traveltime', 
            'studytime', 'famrel', 'freetime', 'goout', 'health']:
    get_stat_dif(col)

# Вывод: имеют статистически значимые различия для колонок sex, address, Mjob, schoolsup, studytime, goout


# In[98]:


# 9. оставляем в модели только те переменные, которые могут влиять на переменную score
math_for_model = math2.loc[:, ['age', 'sex', 'address', 'Mjob',
                               'Medu', 'schoolsup', 'studytime', 
                               'failures', 'absences', 'score']]
math_for_model.head()


# In[99]:


# записываем данные в excel
# math_for_model.to_excel("Project2.xlsx",sheet_name='Лист1')

