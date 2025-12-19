import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


wavelengths = {
    "R": 650e-9,   # красный
    "G": 550e-9,   # зелёный
    "B": 450e-9    # синий
}


#        ПАРАМЕТРЫ ОПТИЧЕСКОЙ СИСТЕМЫ НЕРЕАЛИСТИЧНЫЕ
pinhole_radius = 4e-3     # радиус отверстия камеры-обскура, м
z = 100e-3                # расстояние до сенсора, м
N = 2048                  # размер расчётной сетки
L = 10e-3                 # физический размер области, м
dx = L / N                # размер пикселя

'''
#        ПАРАМЕТРЫ ОПТИЧЕСКОЙ СИСТЕМЫ РЕАЛИСТРИЧНЫЕ
pinhole_radius = 0.4e-3   # радиус отверстия камеры-обскура, м
z = 20e-3                 # расстояние до сенсора, м
N = 2048                  # размер расчётной сетки
L = 10e-3                 # физический размер области, м
dx = L / N                # размер пикселя
'''

image_file = "Путь до фотографии"

# Загружаем изображение и приводим к размеру NxN
img = Image.open(image_file).convert("RGB")
img = img.resize((N, N))
obj_rgb = np.array(img) / 255.0   # нормировка яркости 0..1


#          КООРДИНАТЫ В ПРОСТРАНСТВЕ
x = np.linspace(-L/2, L/2, N)
y = np.linspace(-L/2, L/2, N)
X, Y = np.meshgrid(x, y)
R = np.sqrt(X**2 + Y**2)
pinhole = (R < pinhole_radius)


#      ФУНКЦИЯ РАСПРОСТРАНЕНИЯ ПО ФРЕНЕЛЮ
def fresnel_propagate(u_in, wavelength):
    """
    u_in       — комплексное поле на отверстии
    wavelength — длина волны
    """

    k = 2 * np.pi / wavelength

    # Фазовый множитель Френеля
    H = np.exp(1j * k / (2 * z) * (X**2 + Y**2))

    # Поле сразу за отверстием
    U = u_in * H

    # Применяем Фурье-преобразование, приведение волн в простраство частот и центрирование спектра
    U_fft = np.fft.fftshift(
                np.fft.fft2(
                    np.fft.fftshift(U) #(fast fourier transform)
                )
            )

    # Нормировочный коэффициент
    const = np.exp(1j * k * z) / (1j * wavelength * z)

    return const * U_fft


#     СЧИТАЕМ ИЗОБРАЖЕНИЕ ДЛЯ КАЖДОГО ЦВЕТА
sensor_rgb = np.zeros((N, N, 3))

for i, (color, wl) in enumerate(wavelengths.items()):

    # Амплитуда волны пропорциональна яркости соответствующего канала
    u_obj = obj_rgb[:, :, i].astype(complex)

    # Применяем отверстие
    u_pinhole = u_obj * pinhole

    # Распространяем волну
    u_sensor = fresnel_propagate(u_pinhole, wl)

    # Интенсивность = квадрат модуля
    sensor_rgb[:, :, i] = np.abs(u_sensor)**2



#       НОРМИРОВКА ДЛЯ ОТОБРАЖЕНИЯ

sensor_rgb /= sensor_rgb.max()

#             ВИЗУАЛИЗАЦИЯ

plt.figure(figsize=(15, 4))

plt.subplot(1, 3, 1)
plt.title("Исходный объект")
plt.imshow(obj_rgb)
plt.axis("off")

plt.subplot(1, 3, 2)
plt.title("Отверстие камеры-обскуры")
plt.imshow(pinhole, cmap="gray")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.title("Изображение на сенсоре (RGB)")
plt.imshow(sensor_rgb)
plt.axis("off")

plt.tight_layout()


#      ОЦЕНКА ЭНЕРГЕТИКИ КАМЕРЫ-ОБСКУРА

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

    print("\n===== ОЦЕНКА ЭНЕРГЕТИКИ КАМЕРЫ-ОБСКУРЫ =====")
    print(f"Освещённость объекта: {illuminance_lux} лк")
    print(f"Радиус отверстия: {pinhole_radius*1e3:.2f} мм")
    print(f"Выдержка: {exposure_time*1e3:.0f} мс")
    print(f"Оптическая мощность в камере: {optical_power:.3e} Вт")
    print(f"Энергия за выдержку: {energy:.3e} Дж")
    print(f"Энергия на сенсоре: {effective_energy:.3e} Дж")
    print(f"Плотность энергии на сенсоре: {energy_density:.3e} Дж/м²")
    print("===========================================\n")


# ВЫЗОВ
print_light_estimate()
plt.show()