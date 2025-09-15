A_t = 
\begin{cases}
\text{random action (exploration)}, & \text{with probability } \varepsilon, \\
\arg\max_a Q(a), & \text{with probability } 1 - \varepsilon,
\end{cases}