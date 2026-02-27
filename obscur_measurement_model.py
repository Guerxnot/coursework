import numpy as np

wavelengths = {
    "R": 650e-9,
    "G": 550e-9,
    "B": 450e-9
}

MY_CONSTANT = 5 # СО СКОЛЬКИ ТОЧЕК СОБИРАЕТ СВЕТ КАЖДЫЙ ПИКСЕЛЬ

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
# Геометрия
# ==================================================
object_angle = 2 * np.arctan((object_size/2) / R_object) * 180 / np.pi

# Плоская координатная сетка объекта
x = np.linspace(-L/2, L/2, N)
y = np.linspace(-L/2, L/2, N)
X, Y = np.meshgrid(x, y)
R_xy = np.sqrt(X**2 + Y**2)

# Угловые координаты сенсора
# создаём массив углов по горизонтали от -половины до +половины угла сенсора (в радианах)
theta_x = np.linspace(-sensor_angular_width/2, sensor_angular_width/2, N) * np.pi/180
theta_y = np.linspace(-sensor_angular_width/2, sensor_angular_width/2, N) * np.pi/180
#  делаем из двух списков большую таблицу N×N, где в каждой клетке — пара углов (THETA_X и THETA_Y)
THETA_X, THETA_Y = np.meshgrid(theta_x, theta_y)

#  считаем реальное расстояние от центра оси до точки на сфере (не по прямой, а по кривой поверхности)
rho = sensor_radius * np.sin(np.sqrt(THETA_X**2 + THETA_Y**2))

# ================================================
# Функция распространения Френеля
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
# Входная функция
# =================================================
def object_function_3d():
    sigma = 0.1e-3  # радиус гауссова шара 0.1 мм
    amp = np.exp(-(X**2 + Y**2) / (2*sigma**2))  # яркость в каждой точке
    phase = np.exp(1j * 20 * (X + Y))
    return amp * phase

# =================================================
# ОПЕРАТОР ИЗМЕРЕНИЯ (дискретизация)
# =================================================
def measure_sensor(intensity, pixel_size):
    """
    Интеграл по пикселю / площадь пикселя
    """
    block = int(pixel_size / dx)
    if block <= 1:
        return intensity

    M = intensity.shape[0] // block
    intensity = intensity[:M*block, :M*block]
    m = intensity.reshape(M, block, M, block).mean(axis=(1,3))
    return m

# =================================================
# ОСНОВНОЙ РАСЧЁТ ИЗМЕРЕНИЯ
# =================================================
def run_measurement():

    sensor_signal = {}
    field_before_measure = {}

    # Исходная функция на объекте (сохраняем для сравнения)
    u_obj_base = object_function_3d()
    obj_amplitude = np.abs(u_obj_base)  # яркость на объекте
    obj_amplitude /= np.max(obj_amplitude) + 1e-100  # нормируем на 1.0

    for color, wl in wavelengths.items():
        u_obj = u_obj_base.copy()
        k = 2 * np.pi / wl  # волновое число

        # добавляем начальную кривизну волны от объекта
        u_obj *= np.exp(1j * (np.pi / (wl * R_object)) * (X ** 2 + Y ** 2))

        # распространение до плоскости отверстия
        u_at_pinhole = fresnel_propagate(u_obj, wl, R_object)

        # маска отверстия
        pinhole = (R_xy < pinhole_radius).astype(complex)
        u_after = u_at_pinhole * pinhole

        # распространение до сферического сенсора
        u_sensor = fresnel_propagate(u_after, wl, sensor_radius, rho_grid=rho)

        intens = np.abs(u_sensor)**2
        intens /= np.max(intens) + 1e-100  # нормируем до дискретизации
        field_before_measure[color] = intens

        measured = measure_sensor(intens, pixel_size=MY_CONSTANT*dx)
        measured /= np.max(measured) + 1e-100  # нормируем после
        sensor_signal[color] = measured

    return field_before_measure, sensor_signal, obj_amplitude

# =================================================
if __name__ == "__main__":

    before, after, obj_amplitude = run_measurement()

    print("Входная функция объекта: гауссов шар радиусом 0.1 мм (для примера)")
    print(f"Максимальная яркость на объекте: {np.max(obj_amplitude):.6f}")
    print(f"Яркость в центре объекта: {obj_amplitude[N//2, N//2]:.6f}\n")

    print("После прохождения через систему (до дискретизации):")
    print(f"- Размер поля: {before['R'].shape[0]}×{before['R'].shape[1]}")
    print(f"- Максимальная яркость: {np.max(before['R']):.6f}")
    print(f"- Яркость в центре: {before['R'][N//2, N//2]:.6f}")

    shape_after = after['R'].shape
    print(f"\nПосле дискретизации (сенсор с пикселями {MY_CONSTANT}×{MY_CONSTANT}):")
    print(f"- Размер: ≈{shape_after[0]}×{shape_after[1]}")
    print(f"- Максимальная яркость: {np.max(after['R']):.6f}")
    print(f"- Яркость в центре: {after['R'][shape_after[0]//2, shape_after[1]//2]:.6f}")

