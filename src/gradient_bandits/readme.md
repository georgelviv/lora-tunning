Softmax

\pi(a) = \frac{e^{H(a)}}{\sum_{b} e^{H(b)}}


Preferences for current action

H_{t+1}(A_t) = H_t(A_t) + \alpha (R_t - \bar{R}_t) (1 - \pi(A_t))

Preferences for other actions

H_{t+1}(a) = H_t(a) - \alpha (R_t - \bar{R}_t)\,\pi(a), \quad a \neq A_t


## Chart

softmax(Gradient(A_{primary})) \cup random(Q(A_{secondary}))

softmax(Gradient(A_{primary})) \cup argmax(Q(A_{secondary}))