import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# 案例7: Robertson刚性系统
def robertson(t, y):
    """
    Robertson化学反应问题（经典刚性系统）
    """
    y1, y2, y3 = y
    dy1dt = -0.04 * y1 + 1e4 * y2 * y3
    dy2dt = 0.04 * y1 - 1e4 * y2 * y3 - 3e7 * y2**2
    dy3dt = 3e7 * y2**2
    return [dy1dt, dy2dt, dy3dt]

# 初始条件
y0 = [1.0, 0.0, 0.0]
t_span = (0, 1e5)
t_eval = np.logspace(-6, 5, 500)  # 对数时间网格

# 比较不同求解器
methods = ['RK45', 'BDF', 'Radau']
colors = ['b', 'r', 'g']
labels = ['RK45 (非刚性)', 'BDF (刚性)', 'Radau (刚性)']

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

for method, color, label in zip(methods, colors, labels):
    try:
        sol = solve_ivp(robertson, t_span, y0, method=method, 
                       t_eval=t_eval, rtol=1e-6, atol=1e-8)
        
        # y1随时间变化
        axes[0, 0].semilogx(sol.t, sol.y[0], color=color, 
                            linewidth=2, label=label, alpha=0.7)
        
        # y2随时间变化（对数坐标）
        axes[0, 1].loglog(sol.t, sol.y[1], color=color, 
                         linewidth=2, label=label, alpha=0.7)
        
        # y3随时间变化
        axes[1, 0].semilogx(sol.t, sol.y[2], color=color, 
                           linewidth=2, label=label, alpha=0.7)
        
        # 守恒量 (y1 + y2 + y3应该等于1)
        conservation = sol.y[0] + sol.y[1] + sol.y[2]
        axes[1, 1].semilogx(sol.t, np.abs(conservation - 1), color=color,
                           linewidth=2, label=label, alpha=0.7)
        
        print(f"{method}: 函数评估次数 = {sol.nfev}")
        
    except Exception as e:
        print(f"{method} 失败: {str(e)}")

axes[0, 0].set_xlabel('时间 t')
axes[0, 0].set_ylabel('y₁(t)')
axes[0, 0].set_title('组分 y₁')
axes[0, 0].legend()
axes[0, 0].grid(True)

axes[0, 1].set_xlabel('时间 t')
axes[0, 1].set_ylabel('y₂(t)')
axes[0, 1].set_title('组分 y₂ (对数尺度)')
axes[0, 1].legend()
axes[0, 1].grid(True)

axes[1, 0].set_xlabel('时间 t')
axes[1, 0].set_ylabel('y₃(t)')
axes[1, 0].set_title('组分 y₃')
axes[1, 0].legend()
axes[1, 0].grid(True)

axes[1, 1].set_xlabel('时间 t')
axes[1, 1].set_ylabel('|y₁+y₂+y₃-1|')
axes[1, 1].set_title('守恒量误差')
axes[1, 1].legend()
axes[1, 1].grid(True)

plt.tight_layout()
plt.show()

print("\n注意：刚性求解器(BDF, Radau)在这类问题上效率远高于RK45")