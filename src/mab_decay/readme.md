
# Epsilon decay
begin{equation}
\epsilon_{t+1} = \max \big( \epsilon_{\min}, \, \alpha \cdot \epsilon_t \big)
\end{equation}



# Reward stationary update
\bar{R}_n(a) = \frac{1}{n} \sum_{i=1}^{n} R_i(a)


# Reward exponential

Q_{t+1}(a) = (1 - \alpha) Q_t(a) + \alpha R_t
