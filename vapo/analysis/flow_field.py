import numpy as np
import matplotlib.pyplot as plt
import affordance.utils.flowlib as flowlib
import cv2

n_values = 20
# factor = np.array([x**2 for x in range(n_values)])
factor = 1
x, y = np.meshgrid(np.linspace(-1, 1, n_values) * factor,
                   np.linspace(-1, 1, n_values) * factor)

u = x
v = y
plt.rcParams["figure.figsize"] = (7, 7)

flow = np.stack([u, v]).transpose((1, 2, 0))
flow_img = flowlib.flow_to_image(flow)

w = n_values//2
circle = cv2.circle(np.zeros_like(flow_img), (w, w), w,
                    (255, 255, 255), -1)

white = np.ones_like(circle) * 255
flow_img[circle == 0] = white[circle == 0]

# cv2.imshow('flow', flow_img)
# cv2.waitKey(0)
widths = np.linspace(0, 3, x.size)

plt.quiver(x, y, u, v, headwidth=5, linewidths=widths)
plt.axis('off')
# plt.show()
# cv2.imwrite("/mnt/484A4CAB4A4C9798/GoogleDrive/Maestria-Drive/color.png", flow_img)
plt.savefig("/mnt/484A4CAB4A4C9798/GoogleDrive/Maestria-Drive/flow.png",
            bbox_inches="tight", pad_inches=0)
