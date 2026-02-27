import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

wavelengths = {
    "R": 650e-9,
    "G": 550e-9,
    "B": 450e-9
}

# =================================================
#  ПАРАМЕТРЫ ОПТИЧЕСКОЙ СИСТЕМЫ
# =================================================
pinhole_radius = 0.05e-3      # радиус отверстия, м
sensor_radius = 0.5e-3        # радиус сферического сенсора, м
N = 2048
object_size = 0.5e-3          # физический размер объекта, м
L = object_size
dx = L / N                    # размер пикселя
R_object = 2e-3               # расстояние до объекта, м
sensor_angular_width = 30.0   # угловой размер сенсора, градусы

# ==================================================
#  Геометрия
# ==================================================
object_angle = 2 * np.arctan((object_size/2) / R_object) * 180 / np.pi

print(f"\n=== ПРОВЕРКА ГЕОМЕТРИИ ===")
print(f"Размер объекта: {object_size*1000:.1f} мм")
print(f"Расстояние до объекта: {R_object*1000:.1f} мм")
print(f"Угол обзора сенсора: {sensor_angular_width:.1f}°")
print(f"Угол, под которым виден объект: {object_angle:.1f}°")
if sensor_angular_width >= object_angle:
    print(" Объект ПОЛНОСТЬЮ помещается в кадр")
else:
    print(" Объект НЕ помещается в кадр!")
print("="*50)

# ===============================================
#  Загрузка объекта
# ================================================
image_file = "Фото и путь до него"
img = Image.open(image_file).convert("RGB")
img = img.resize((N, N))
obj_rgb = np.array(img) / 255.0 # делаем яркость от 0 до 1, нормировка

# Плоская координатная сетка объекта
x = np.linspace(-L/2, L/2, N)
y = np.linspace(-L/2, L/2, N)
X, Y = np.meshgrid(x, y)  # X и Y — матрицы N x N, где каждая точка имеет координаты
R_xy = np.sqrt(X**2 + Y**2)

#  Угловые координаты сенсора

# создаём массив углов по горизонтали от -половины до +половины угла сенсора (в радианах)
theta_x = np.linspace(-sensor_angular_width/2, sensor_angular_width/2, N) * np.pi/180
theta_y = np.linspace(-sensor_angular_width/2, sensor_angular_width/2, N) * np.pi/180
#  делаем из двух списков большую таблицу N×N, где в каждой клетке — пара углов (THETA_X и THETA_Y)
THETA_X, THETA_Y = np.meshgrid(theta_x, theta_y)

#  считаем реальное расстояние от центра оси до точки на сфере (не по прямой, а по кривой поверхности)
rho = sensor_radius * np.sin(np.sqrt(THETA_X**2 + THETA_Y**2))
print(f"Сферический сенсор: радиус = {sensor_radius*1e3:.2f} мм, угол = {sensor_angular_width}°")


# ================================================
#  Функция распространения Френеля
# ================================================
def fresnel_propagate(u_in, wavelength, z, rho_grid=None):
    """
    u_in     — комплексное поле
    z        — расстояние распространения
    rho_grid — поперечное расстояние от оси
    """
    if rho_grid is None:
        rho2 = X**2 + Y**2
    else:
        rho2 = rho_grid**2

    k = 2 * np.pi / wavelength   # волновое число
    phase = (np.pi / (wavelength * z)) * rho2

    # Поворачивает фазу волны, 1j - мнимая единица
    H = np.exp(1j * phase)

    # Поворот фазы к полю света
    U = u_in * H

    # (fast fourier transform)
    # fft2 — Фурье преобразование, превращает изображение в набор частот
    # fftshift — переставляет пиксели, чтобы центр был в середине
    U_fft = np.fft.fftshift(np.fft.fft2(np.fft.fftshift(U)))

    # Константа распространения, общая фаза + масштаб
    const = np.exp(1j * k * z) / (1j * wavelength * z)

    return const * U_fft


# =================================================
#  Основной расчёт
# =================================================
sensor_rgb_sphere = np.zeros((N, N, 3), dtype=float)

for i, (color, wl) in enumerate(wavelengths.items()):
    print(f"Расчёт канала {color} ({wl*1e9:.0f} нм)...")

    # берём один цвет из изображения
    # и делаем его комплексным
    u_obj = obj_rgb[..., i].astype(complex)

    k = 2 * np.pi / wl   # волновое число

    # добавляем начальную кривизну волны от объекта
    u_obj *= np.exp(1j * (np.pi / (wl * R_object)) * (X**2 + Y**2))

    # распространение до плоскости отверстия
    u_at_pinhole = fresnel_propagate(u_obj, wl, R_object)

    # маска отверстия
    pinhole = (R_xy < pinhole_radius).astype(complex)
    u_after = u_at_pinhole * pinhole

    # распространение до сферического сенсора
    u_sensor = fresnel_propagate(u_after, wl, sensor_radius, rho_grid=rho)

    # интенсивность
    intens = np.abs(u_sensor)**2
    intens /= np.max(intens) + 1e-100 # нормализация на канал
    sensor_rgb_sphere[..., i] = intens

# глобальная нормализация для RGB
max_all = np.percentile(sensor_rgb_sphere, 99.5) # игнорируем редкие всплески шума
sensor_rgb_sphere /= max_all + 1e-100
sensor_rgb_sphere = np.clip(sensor_rgb_sphere, 0, 1)
def print_light_estimate():
    illuminance_lux = 10000     # освещённость объекта (люкс)
    exposure_time = 0.1         # выдержка (сек)
    sensor_size = 5e-3          # размер сенсора 5 мм
    efficiency = 0.3            # общая предполагаемая эффективность системы

    pinhole_area = np.pi * pinhole_radius**2
    sensor_area = sensor_size**2

    luminous_flux = illuminance_lux * pinhole_area        # люмен
    optical_power = luminous_flux / 683                   # Вт
    # 683 лм/Вт - световая эффективность для зеленого света
    energy = optical_power * exposure_time                # Дж
    effective_energy = energy * efficiency                # Дж
    energy_density = effective_energy / sensor_area       # Дж/м²

    print("\n===== ОЦЕНКА ЭНЕРГЕТИКИ КАМЕРЫ ОБСКУРА =====")
    print(f"Освещённость объекта: {illuminance_lux} лк")
    print(f"Радиус отверстия: {pinhole_radius*1e3:.2f} мм")
    print(f"Выдержка: {exposure_time*1e3:.0f} мс")
    print(f"Оптическая мощность в камере: {optical_power:.3e} Вт")
    print(f"Энергия за выдержку: {energy:.3e} Дж")
    print(f"Энергия на сенсоре: {effective_energy:.3e} Дж")
    print(f"Плотность энергии на сенсоре: {energy_density:.3e} Дж/м²")
    print("Необходимая энергия для нормальной работы современных сенсоров: 10⁻⁶ — 10⁻⁴ Дж/м²\n")

# ===========================================
#  Визуализация
# ===========================================
plt.figure(figsize=(20, 5))

plt.subplot(1, 4, 1)
plt.title(f"Объект (расст. {R_object*1e3:.1f} мм)")
plt.imshow(obj_rgb)
plt.axis("off")

plt.subplot(1, 4, 2)
plt.title(f"Отверстие (R={pinhole_radius*1e3:.2f} мм)")
plt.imshow(pinhole.real, cmap="gray")
plt.axis("off")

plt.subplot(1, 4, 3)
plt.title(f"Сферический сенсор\n{sensor_angular_width}° × {sensor_angular_width}°")
plt.imshow(sensor_rgb_sphere,
           extent=[-sensor_angular_width/2, sensor_angular_width/2,
                   -sensor_angular_width/2, sensor_angular_width/2])
plt.xlabel("Угол X (°)")
plt.ylabel("Угол Y (°)")
plt.colorbar(shrink=0.6)

ax = plt.subplot(1, 4, 4, projection='3d')
ax.set_title("Проекция на сферу", pad=15)
step = 30
theta = np.sqrt(THETA_X**2 + THETA_Y**2)
phi = np.arctan2(THETA_Y, THETA_X)
X_plot = sensor_radius * np.sin(theta[::step, ::step]) * np.cos(phi[::step, ::step])
Y_plot = sensor_radius * np.sin(theta[::step, ::step]) * np.sin(phi[::step, ::step])
Z_plot = sensor_radius * np.cos(theta[::step, ::step])
colors = sensor_rgb_sphere[::step, ::step, :]
colors = np.clip(colors, 0, 1)
ax.scatter(X_plot, Y_plot, Z_plot,
           c=colors.reshape(-1, 3),
           s=12,
           alpha=0.92,
           edgecolor='none')
ax.scatter(0, 0, 0, color='red', s=140, marker='o',
           edgecolor='darkred', linewidth=1.5,
           label='Отверстие (центр)')
ax.set_xlabel('X (м)')
ax.set_ylabel('Y (м)')
ax.set_zlabel('Z (м)')
ax.set_box_aspect([1,1,1])
ax.view_init(elev=20, azim=-60)
ax.legend(loc='upper right', fontsize=9)
ax.grid(True, alpha=0.15)

# =================================================================
# ОТДЕЛЬНАЯ КРУПНАЯ 3D ВИЗУАЛИЗАЦИЯ СФЕРИЧЕСКОГО СЕНСОРА
# =================================================================
fig2 = plt.figure(figsize=(14, 7))
ax2 = fig2.add_subplot(111, projection='3d')
ax2.set_title("СФЕРИЧЕСКИЙ СЕНСОР КАМЕРЫ-ОБСКУРЫ\n(3D визуализация с нанесенным изображением)",
              fontsize=16, fontweight='bold', pad=20)

# Параметры отображения
step_3d = 35  # Шаг прореживания для скорости

# Преобразование угловых координат в сферические
theta_3d = np.sqrt(THETA_X ** 2 + THETA_Y ** 2)  # полярный угол от оси
phi_3d = np.arctan2(THETA_Y, THETA_X)  # азимутальный угол

# Декартовы координаты точек на сфере (с нанесенным изображением)
X_sensor_3d = sensor_radius * np.sin(theta_3d[::step_3d, ::step_3d]) * np.cos(phi_3d[::step_3d, ::step_3d])
Y_sensor_3d = sensor_radius * np.sin(theta_3d[::step_3d, ::step_3d]) * np.sin(phi_3d[::step_3d, ::step_3d])
Z_sensor_3d = sensor_radius * np.cos(theta_3d[::step_3d, ::step_3d])

# Цвета с изображения сенсора
colors_3d = sensor_rgb_sphere[::step_3d, ::step_3d, :]
colors_3d = np.clip(colors_3d, 0, 1)

# Рисуем сферический сенсор с нанесенным изображением
scatter = ax2.scatter(X_sensor_3d, Y_sensor_3d, Z_sensor_3d,
                      c=colors_3d.reshape(-1, 3),
                      s=25, alpha=0.95, edgecolor='none',
                      label='Сферический сенсор\nс изображением')

# Рисуем отверстие (центр сферы)
ax2.scatter(0, 0, 0,
            color='red', s=300, marker='o',
            edgecolor='darkred', linewidth=2,
            alpha=0.9, label=f'Отверстие\nR = {pinhole_radius * 1e3:.3f} мм')

# Рисуем оптическую ось
ax2.plot([-sensor_radius * 0.3, sensor_radius * 1.2], [0, 0], [0, 0],
         'k--', alpha=0.5, linewidth=1, label='Оптическая ось')

# Добавляем линии, показывающие направление на объект
# Объект находится на расстоянии R_object в отрицательном направлении Z
obj_z_pos = -R_object
obj_x_pos = 0
obj_y_pos = 0

# Рисуем точку объекта (условно)
ax2.scatter(obj_x_pos, obj_y_pos, obj_z_pos,
            color='blue', s=200, marker='^',
            edgecolor='darkblue', linewidth=2,
            alpha=0.8, label=f'Объект\nна расстоянии {R_object * 1e3:.1f} мм')

# Соединяем объект с отверстием (основной луч)
ax2.plot([obj_x_pos, 0], [obj_y_pos, 0], [obj_z_pos, 0],
         'b-', alpha=0.4, linewidth=1.5)

# Добавляем несколько лучей от объекта к краям сенсора для наглядности
for angle in np.linspace(-sensor_angular_width / 4, sensor_angular_width / 4, 5):
    angle_rad = angle * np.pi / 180
    # Направление на край сенсора
    x_edge = sensor_radius * np.sin(angle_rad)
    z_edge = sensor_radius * np.cos(angle_rad)

    # Луч от объекта через отверстие к сенсору
    ax2.plot([obj_x_pos, 0, x_edge],
             [0, 0, 0],
             [obj_z_pos, 0, z_edge],
             'g-', alpha=0.15, linewidth=1)

# Настройка подписей с расстояниями
# Подпись расстояния объект-отверстие
ax2.text(obj_x_pos, obj_y_pos, obj_z_pos / 2,
         f'{R_object * 1e3:.1f} мм',
         color='blue', fontsize=10, fontweight='bold',
         bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))

# Подпись расстояния отверстие-сенсор
ax2.text(sensor_radius * 0.5, 0, sensor_radius * 0.5,
         f'{sensor_radius * 1e3:.2f} мм',
         color='green', fontsize=10, fontweight='bold',
         bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))

# Подпись угла обзора
angle_text_x = sensor_radius * np.sin(sensor_angular_width / 4 * np.pi / 180) * 1.2
angle_text_z = sensor_radius * np.cos(sensor_angular_width / 4 * np.pi / 180) * 1.2
ax2.text(angle_text_x, 0, angle_text_z,
         f'{sensor_angular_width}°',
         color='purple', fontsize=11, fontweight='bold',
         bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))

# Добавляем прозрачную сферу для визуализации формы
u = np.linspace(0, 2 * np.pi, 40)
v = np.linspace(0, sensor_angular_width / 2 * np.pi / 180, 20)
x_sphere_wire = sensor_radius * np.outer(np.cos(u), np.sin(v))
y_sphere_wire = sensor_radius * np.outer(np.sin(u), np.sin(v))
z_sphere_wire = sensor_radius * np.outer(np.ones(np.size(u)), np.cos(v))

ax2.plot_wireframe(x_sphere_wire, y_sphere_wire, z_sphere_wire,
                   color='gray', alpha=0.15, linewidth=0.5)

# Настройки оформления
ax2.set_xlabel('X (м)', fontsize=12, labelpad=10)
ax2.set_ylabel('Y (м)', fontsize=12, labelpad=10)
ax2.set_zlabel('Z (м)', fontsize=12, labelpad=10)

# Устанавливаем одинаковый масштаб по всем осям
max_range = max(sensor_radius, abs(obj_z_pos)) * 1.2
ax2.set_xlim(-max_range, max_range)
ax2.set_ylim(-max_range, max_range)
ax2.set_zlim(-max_range, max_range)
ax2.set_box_aspect([1, 1, 1])

# Настройка угла обзора
ax2.view_init(elev=25, azim=-65)

# Легенда
ax2.legend(loc='upper left', fontsize=10, framealpha=0.9)

# Сетка
ax2.grid(True, alpha=0.2)

# Заголовок с параметрами
param_text = f"Параметры: r_отв={pinhole_radius * 1e3:.3f}мм | R_объект={R_object * 1e3:.1f}мм | R_сенсор={sensor_radius * 1e3:.2f}мм | Угол={sensor_angular_width}°"
fig2.text(0.5, 0.02, param_text, ha='center', fontsize=11,
          bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
print_light_estimate()
plt.tight_layout()
plt.show()
