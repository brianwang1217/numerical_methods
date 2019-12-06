import numpy as np
import matplotlib.pyplot as plt
import math

''' Finding Waldo '''
min_diff = abs(np.linalg.norm(waldo_image) - np.linalg.norm(group_image[:len(waldo_image),:len(waldo_image[0])]))

for i in range (len(group_image) - len(waldo_image)):
    for j in range (len(group_image[0]) - len(waldo_image[0])):
        if (abs(np.linalg.norm(waldo_image - group_image[i:(i+len(waldo_image)),j:(j+len(waldo_image[0]))]) < min_diff)):
            min_diff = abs(np.linalg.norm(waldo_image - group_image[i:(i+len(waldo_image)),j:(j+len(waldo_image[0]))]))
            x_coord = i
            y_coord = j

top_left = (x_coord, y_coord)
print(top_left, min_diff)
plt.imshow((group_image))
#plt.imshow(waldo_image)

''' Norm Ball '''
def norm(x, p):
    sum = 0.0
    for ele in x:
        sum += (abs(ele) ** p)
    return (sum ** (1/p))

ps = [1, 2, 5, 0.5]
phi = np.linspace(0, 2 * np.pi, 500)

unit_circle = np.array([np.cos(phi), np.sin(phi)])

#fig1
vectors1 = unit_circle * r
plt.figure()
for i in range(len(vectors1[0])):
    x = vectors1[0, i]
    y = vectors1[1, i]
    vectors1[0, i] = r * x / (norm([x, y], ps[0]))
    vectors1[1, i] = r * y / (norm([x, y], ps[0]))

plt.plot(vectors1[0], vectors1[1])
fig_1 = plt.gca()

#fig2
vectors2 = unit_circle * r
plt.figure()
for i in range(len(vectors2[0])):
    vectors2[0, i] = vectors2[0, i] * norm([vectors2[0, i], vectors2[1, i]], ps[1]) ** -1 * r
    vectors2[1, i] = vectors2[1, i] * norm([vectors2[0, i], vectors2[1, i]], ps[1]) ** -1 * r

plt.plot(vectors2[0], vectors2[1])
fig_2 = plt.gca()

#fig3
vectors3 = unit_circle * r
plt.figure()
for i in range(len(vectors3[0])):
    x = vectors3[0, i]
    y = vectors3[1, i]
    vectors3[0, i] = r * x / (norm([x, y], ps[2]))
    vectors3[1, i] = r * y / (norm([x, y], ps[2]))

plt.plot(vectors3[0], vectors3[1])
fig_3 = plt.gca()

#fig4
vectors4 = unit_circle * r
plt.figure()
for i in range(len(vectors4[0])):
    x = vectors4[0, i]
    y = vectors4[1, i]
    vectors4[0, i] = r * x / (norm([x, y], ps[3]))
    vectors4[1, i] = r * y / (norm([x, y], ps[3]))

plt.plot(vectors4[0], vectors4[1])
fig_4 = plt.gca()

fig_1.set_aspect('equal')
fig_2.set_aspect('equal')
fig_3.set_aspect('equal')
fig_4.set_aspect('equal')

#print(vectors1[:10, :10], vectors2[:10, :10], vectors3[:10, :10], vectors4[:10, :10])
plt.plot()
