
# Epsilon decay
begin{equation}
\epsilon_{t+1} = \max \big( \epsilon_{\min}, \, \alpha \cdot \epsilon_t \big)
\end{equation}



# Reward stationary update
\bar{R}_n(a) = \frac{1}{n} \sum_{i=1}^{n} R_i(a)


# Reward exponential

Q_{t+1}(a) = (1 - \alpha) Q_t(a) + \alpha R_t


# Reward stationary
Q_{t}(a) = \frac{R{1}+R{2}+...+R{n}}{n}


## Ідея
1. Беремо рандомну дію A{t}
2. Оцінюємо дію R{t+1}
3. Шукаємо у таблиці Q рядок даної дії A{t} 
4. Якщо рядка нема, тоді додаємо як [A{t}, Count=0, Reward=R{t+1}]
5. Якшо рядок є,
   1. Рахуємо New Reward (Q_{\text{new}}) через середнє арифметичне = Q_{\text{new}} = Q_{\text{old}} + \frac{R_{t-1}-Q_{\text{old}}}{n}
   2. [A{t}, Count=Count+1, Reward=New Reward]
6. Якшо \text{rand}() > \epsilon
   1. Беремо з таблички найкраще A{t} по Reward
7. Якшо ні
   1. Беремо рандомну дію A{t}
8. Повертаємось до пункту 2
