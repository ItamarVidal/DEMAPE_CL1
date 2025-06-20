import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from mpl_toolkits.mplot3d import Axes3D

# Parâmetros fixos
lambda_ = 550e-9  # Comprimento de onda (550 nm, pico para 4000 K)
w0 = 0.004  # Tamanho efetivo do LED (4 mm)
I0 = 1  # Intensidade máxima do LED (normalizada)
d = 6.0  # Distância do anteparo (6 m)
z_led_to_lens = 0.05  # Distância do LED até a lente (5 cm)
d_lens_to_screen = d - z_led_to_lens  # Distância da lente até o anteparo (5.95 m)
n = 1.5  # Índice de refração (PMMA)

# Área desejada de iluminação (para referência)
w_x = 5.0  # Largura do retângulo (5 m)
w_y = 10.0  # Comprimento do retângulo (10 m)
sigma_x = 1.0  # Suavidade das bordas em x (usado apenas para o perfil desejado)
sigma_y = 2.0  # Suavidade das bordas em y (usado apenas para o perfil desejado)

# Grade 1D para otimização e comparação
x_1d = np.linspace(-10, 10, 500)
y_1d = np.linspace(-15, 15, 500)

# Perfil desejado 1D (apenas para comparação)
I_max = 1
I_desired_x = np.ones_like(x_1d) * I_max
mask_left_x = x_1d < -w_x/2
mask_right_x = x_1d > w_x/2
I_desired_x[mask_left_x] = I_max * np.exp(-(x_1d[mask_left_x] + w_x/2)**2 / (2 * sigma_x**2))
I_desired_x[mask_right_x] = I_max * np.exp(-(x_1d[mask_right_x] - w_x/2)**2 / (2 * sigma_x**2))

I_desired_y = np.ones_like(y_1d) * I_max
mask_left_y = y_1d < -w_y/2
mask_right_y = y_1d > w_y/2
I_desired_y[mask_left_y] = I_max * np.exp(-(y_1d[mask_left_y] + w_y/2)**2 / (2 * sigma_y**2))
I_desired_y[mask_right_y] = I_max * np.exp(-(y_1d[mask_right_y] - w_y/2)**2 / (2 * sigma_y**2))

# Função para calcular o perfil colimado
def calculate_collimated_profile(f_col, x, w0, lambda_, I0):
    w_col = (lambda_ * f_col / (np.pi * w0))
    I = I0 * (w0 / w_col)**2 * np.exp(-2 * (x / w_col)**2)
    I_max_val = I.max()
    if I_max_val > 0:
        I = I / I_max_val * I_max
    return I

# Função para calcular o perfil após a lente de Fresnel (usado para otimização inicial)
def calculate_fresnel_profile(f_fresnel, x, I_collimated, d, w_target):
    scale = w_target * f_fresnel / d
    I = I_collimated * (scale / (scale + np.abs(x)))**2
    I_max_val = I.max()
    if I_max_val > 0:
        I = I / I_max_val * I_max
    return I

# Função de custo para otimização
##
##
##
##
##    Deixado propositalmente em branco #########
##
##
##
##

# Otimizar os parâmetros
initial_params = [0.1, 0.1, 0.1]
result = minimize(
    cost_function,
    initial_params,
    args=(x_1d, y_1d, w0, lambda_, I0, I_desired_x, I_desired_y),
    method='Nelder-Mead',
    bounds=[(0.01, 1.0), (0.01, 1.0), (0.01, 1.0)]
)

# Parâmetros otimizados
f_col_opt, f_fresnel_x_opt, f_fresnel_y_opt = result.x
print(f"Distância focal da lente colimadora: f_col = {f_col_opt:.3f} m")
print(f"Distância focal da lente Fresnel (x): f_fresnel_x = {f_fresnel_x_opt:.3f} m")
print(f"Distância focal da lente Fresnel (y): f_fresnel_y = {f_fresnel_y_opt:.3f} m")

# Calcular os perfis otimizados (1D) - Perfil inicial para comparação
I_collimated_x = calculate_collimated_profile(f_col_opt, x_1d, w0, lambda_, I0)
I_collimated_y = calculate_collimated_profile(f_col_opt, y_1d, w0, lambda_, I0)
I_optimized_x = calculate_fresnel_profile(f_fresnel_x_opt, x_1d, I_collimated_x, d, w_x)
I_optimized_y = calculate_fresnel_profile(f_fresnel_y_opt, y_1d, I_collimated_y, d, w_y)

# Plotar os perfis de intensidade (1D) - Comparação com o desejado
plt.figure(figsize=(10, 6))
plt.plot(x_1d, I_optimized_x, label="Perfil Inicial (x)", color="green")
plt.plot(x_1d, I_desired_x, label="Perfil Desejado (x)", color="orange", linestyle="--")
plt.xlabel("Posição X (m)")
plt.ylabel("Intensidade (normalizada)")
plt.legend()
plt.grid()
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(y_1d, I_optimized_y, label="Perfil Inicial (y)", color="green")
plt.plot(y_1d, I_desired_y, label="Perfil Desejado (y)", color="orange", linestyle="--")
plt.xlabel("Posição Y (m)")
plt.ylabel("Intensidade (normalizada)")
plt.legend()
plt.grid()
plt.show()

# --- Projeto da Lente de Fresnel com Ranhuras Bidimensionais ---
lens_size_x = w0 * 15  # Tamanho da lente em x (15x o tamanho do LED)
lens_size_y = w0 * 15  # Tamanho da lente em y (15x o tamanho do LED)
num_grooves = 100  # Número de ranhuras
groove_width_x = lens_size_x / num_grooves
groove_width_y = lens_size_y / num_grooves

# Coordenadas 2D para a lente inteira
x_lens = np.linspace(-lens_size_x/2, lens_size_x/2, 100)
y_lens = np.linspace(-lens_size_y/2, lens_size_y/2, 100)
X_lens, Y_lens = np.meshgrid(x_lens, y_lens)

# Função para calcular o perfil da lente de Fresnel com ranhuras bidimensionais
def fresnel_grooves_2d(f_fresnel_x, f_fresnel_y, x, y, num_grooves, groove_width_x, groove_width_y, n):
    z = np.zeros_like(x)
    for i in range(num_grooves):
        for j in range(num_grooves):
            x_min = -lens_size_x/2 + i * groove_width_x
            x_max = -lens_size_x/2 + (i + 1) * groove_width_x
            y_min = -lens_size_y/2 + j * groove_width_y
            y_max = -lens_size_y/2 + (j + 1) * groove_width_y
            mask = (x >= x_min) & (x < x_max) & (y >= y_min) & (y < y_max)
            # Fase quadrática em x e y com diferentes distâncias focais
            phi_x = (2 * np.pi / lambda_) * (x[mask]**2 / (2 * f_fresnel_x))
            phi_y = (2 * np.pi / lambda_) * (y[mask]**2 / (2 * f_fresnel_y))
            z[mask] = (lambda_ / (2 * np.pi)) * ((phi_x + phi_y) % (2 * np.pi))
    return z

# Calcular o perfil da lente
Z_lens = fresnel_grooves_2d(f_fresnel_x_opt, f_fresnel_y_opt, X_lens, Y_lens, num_grooves, groove_width_x, groove_width_y, n)

# Plotar a lente de Fresnel (vista 3D)
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X_lens, Y_lens, Z_lens, cmap='viridis', edgecolor='none')
ax.set_xlabel("Posição X (m)")
ax.set_ylabel("Posição Y (m)")
ax.set_zlabel("Espessura (m)")
ax.set_title("Lente de Fresnel com Ranhuras Bidimensionais (Vista 3D)")
plt.show()

# Plotar a lente de Fresnel (vista 2D - mapa de calor)
plt.figure(figsize=(8, 6))
plt.imshow(Z_lens, extent=(-lens_size_x/2, lens_size_x/2, -lens_size_y/2, lens_size_y/2), cmap='viridis', origin='lower')
plt.colorbar(label="Espessura (m)")
plt.xlabel("Posição X (m)")
plt.ylabel("Posição Y (m)")
plt.title("Lente de Fresnel com Ranhuras Bidimensionais (Vista 2D)")
plt.show()

# --- Propagação da Luz com Difração (Fresnel Propagation) ---
# Grade 2D para o plano da lente (maior resolução)
x_lens_fine = np.linspace(-lens_size_x/2, lens_size_x/2, 500)
y_lens_fine = np.linspace(-lens_size_y/2, lens_size_y/2, 500)
X_lens_fine, Y_lens_fine = np.meshgrid(x_lens_fine, y_lens_fine)

# Campo inicial do LED (fonte gaussiana) no plano do LED (z = 0)
k = 2 * np.pi / lambda_
field_led = np.sqrt(I0) * np.exp(-(X_lens_fine**2 + Y_lens_fine**2) / (2 * w0**2))

# Propagação do LED até a lente (z = 0.05 m)
dx = x_lens_fine[1] - x_lens_fine[0]
dy = y_lens_fine[1] - y_lens_fine[0]
k_x = 2 * np.pi * np.fft.fftfreq(len(x_lens_fine), dx)
k_y = 2 * np.pi * np.fft.fftfreq(len(y_lens_fine), dy)
KX, KY = np.meshgrid(k_x, k_y)
propagator_led_to_lens = np.exp(1j * z_led_to_lens * np.sqrt(k**2 - KX**2 - KY**2))
field_fft = np.fft.fft2(field_led)
field_propagated_to_lens_fft = field_fft * propagator_led_to_lens
field_at_lens = np.fft.ifft2(field_propagated_to_lens_fft)

# Fase da lente de Fresnel
phi_x = (k / (2 * f_fresnel_x_opt)) * X_lens_fine**2
phi_y = (k / (2 * f_fresnel_y_opt)) * Y_lens_fine**2
lens_phase = np.exp(1j * (phi_x + phi_y))

# Campo após a lente
field_after_lens = field_at_lens * lens_phase

# Propagação da lente até o anteparo (z = 5.95 m)
propagator_lens_to_screen = np.exp(1j * d_lens_to_screen * np.sqrt(k**2 - KX**2 - KY**2))
field_propagated_fft = np.fft.fft2(field_after_lens) * propagator_lens_to_screen
field_propagated = np.fft.ifft2(field_propagated_fft)

# Intensidade no anteparo
I_propagated = np.abs(field_propagated)**2
I_max_val = I_propagated.max()
if I_max_val > 0:
    I_propagated = I_propagated / I_max_val * I_max

# Grade 2D para o anteparo (ajustada para cobrir a área desejada)
x_anteparo = np.linspace(-3, 3, 500)  # 5 m de largura com margem
y_anteparo = np.linspace(-6, 6, 500)  # 10 m de comprimento com margem
X_anteparo, Y_anteparo = np.meshgrid(x_anteparo, y_anteparo)

# --- Visualização 2D da Propagação da Luz (Plano x-z) com Ray Tracing ---
num_rays = 40  # Número de raios
x_rays = np.linspace(-lens_size_x/2, lens_size_x/2, num_rays)  # Posições iniciais dos raios
z_rays = np.linspace(0, d, 100)  # Posições ao longo do eixo z

# Calcular os ângulos de refração para cada raio (aproximação geométrica para visualização)
z_lens = z_led_to_lens  # Posição da lente ao longo de z (5 cm)
angles_x = np.zeros_like(x_rays)
for i, x_start in enumerate(x_rays):
    theta_i = np.arctan(x_start / f_fresnel_x_opt)
    theta_r = np.arcsin(np.sin(theta_i) / n)
    angles_x[i] = theta_r

# Traçar os raios no plano x-z
plt.figure(figsize=(12, 6))
for i in range(num_rays):
    x_start = x_rays[i]
    x_pre_lens = x_start + (z_rays[z_rays <= z_lens] / z_lens) * (x_start - x_start)
    plt.plot(x_pre_lens, z_rays[z_rays <= z_lens], color="yellow", alpha=0.5)
    x_post_lens = x_start + (z_rays[z_rays >= z_lens] - z_lens) * np.tan(angles_x[i])
    plt.plot(x_post_lens, z_rays[z_rays >= z_lens], color="yellow", alpha=0.5)

# Plotar a lente de Fresnel (perfil em x-z)
r = np.linspace(-lens_size_x/2, lens_size_x/2, 500)
z1_x = fresnel_grooves_2d(f_fresnel_x_opt, f_fresnel_y_opt, r, np.zeros_like(r), num_grooves, groove_width_x, groove_width_y, n)
z2_x = -z1_x
plt.plot(r, z1_x + z_lens, color="blue", label="Lente Fresnel")
plt.plot(r, z2_x + z_lens, color="blue")

plt.axvline(x=0, ymin=0, ymax=0.01, color="red", label="LED")
plt.axhline(y=d, xmin=-0.5, xmax=0.5, color="gray", label="Anteparo")
plt.xlabel("Posição X (m)")
plt.ylabel("Distância Z (m)")
plt.title("Propagação da Luz (Plano x-z)")
plt.legend()
plt.grid()
plt.show()

# --- Visualização 2D da Propagação da Luz (Plano y-z) com Ray Tracing ---
y_rays = np.linspace(-lens_size_y/2, lens_size_y/2, num_rays)  # Posições iniciais dos raios
angles_y = np.zeros_like(y_rays)
for i, y_start in enumerate(y_rays):
    theta_i = np.arctan(y_start / f_fresnel_y_opt)
    theta_r = np.arcsin(np.sin(theta_i) / n)
    angles_y[i] = theta_r

# Traçar os raios no plano y-z
plt.figure(figsize=(12, 6))
for i in range(num_rays):
    y_start = y_rays[i]
    y_pre_lens = y_start + (z_rays[z_rays <= z_lens] / z_lens) * (y_start - y_start)
    plt.plot(y_pre_lens, z_rays[z_rays <= z_lens], color="yellow", alpha=0.5)
    y_post_lens = y_start + (z_rays[z_rays >= z_lens] - z_lens) * np.tan(angles_y[i])
    plt.plot(y_post_lens, z_rays[z_rays >= z_lens], color="yellow", alpha=0.5)

# Plotar a lente de Fresnel (perfil em y-z)
r = np.linspace(-lens_size_y/2, lens_size_y/2, 500)
z1_y = fresnel_grooves_2d(f_fresnel_x_opt, f_fresnel_y_opt, np.zeros_like(r), r, num_grooves, groove_width_x, groove_width_y, n)
z2_y = -z1_y
plt.plot(r, z1_y + z_lens, color="blue", label="Lente Fresnel")
plt.plot(r, z2_y + z_lens, color="blue")

plt.axvline(x=0, ymin=0, ymax=0.01, color="red", label="LED")
plt.axhline(y=d, xmin=-0.5, xmax=0.5, color="gray", label="Anteparo")
plt.xlabel("Posição Y (m)")
plt.ylabel("Distância Z (m)")
plt.title("Propagação da Luz (Plano y-z)")
plt.legend()
plt.grid()
plt.show()

# --- Visualização 2D da Iluminação no Anteparo (Perfil Real com Difração) ---
plt.figure(figsize=(8, 6))
plt.imshow(I_propagated, extent=(-3, 3, -6, 6), cmap="hot", origin="lower")
plt.colorbar(label="Intensidade (normalizada)")
plt.xlabel("Posição X (m)")
plt.ylabel("Posição Y (m)")
plt.title("Iluminação Real no Anteparo (2D) - 6 m de Distância")
plt.show()

# --- Visualização 3D da Iluminação Real no Anteparo (no final) ---
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(X_anteparo, Y_anteparo, I_propagated, cmap='hot', edgecolor='none')
ax.set_xlabel("Posição X (m)")
ax.set_ylabel("Posição Y (m)")
ax.set_zlabel("Intensidade (normalizada)")
ax.set_title("Iluminação Real no Anteparo (3D) - 6 m de Distância")
fig.colorbar(surf, ax=ax, label="Intensidade (normalizada)")
plt.show()
