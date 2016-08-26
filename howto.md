http://stackoverflow.com/questions/8827016/matplotlib-savefig-in-jpeg-format
import Image
import matplotlib.pyplot as plt
plt.plot(range(10))
plt.savefig('testplot.png')
Image.open('testplot.png').save('testplot.jpg','JPEG')
