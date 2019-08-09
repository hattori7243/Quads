from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import imageio,os
from collections import Counter
import heapq
import sys
import os
import re

MODE_RECTANGLE = 1
MODE_ELLIPSE = 2
MODE_ROUNDED_RECTANGLE = 3

MODE = MODE_RECTANGLE
ITERATIONS = 1024
LEAF_SIZE = 4
PADDING = 1
FILL_COLOR = (0, 0, 0)
SAVE_FRAMES = False
ERROR_RATE = 0.5
AREA_POWER = 0.25
OUTPUT_SCALE = 1

iter={'start':0,'end':0,'step':0}


def weighted_average(hist):
    total = sum(hist)
    value = sum(i * x for i, x in enumerate(hist)) / total
    error = sum(x * (value - i) ** 2 for i, x in enumerate(hist)) / total
    error = error ** 0.5
    return value, error

def color_from_histogram(hist):
    r, re = weighted_average(hist[:256])
    g, ge = weighted_average(hist[256:512])
    b, be = weighted_average(hist[512:768])
    e = re * 0.2989 + ge * 0.5870 + be * 0.1140
    return (r, g, b), e

def rounded_rectangle(draw, box, radius, color):
    l, t, r, b = box
    d = radius * 2
    draw.ellipse((l, t, l + d, t + d), color)
    draw.ellipse((r - d, t, r, t + d), color)
    draw.ellipse((l, b - d, l + d, b), color)
    draw.ellipse((r - d, b - d, r, b), color)
    d = radius
    draw.rectangle((l, t + d, r, b - d), color)
    draw.rectangle((l + d, t, r - d, b), color)

class Quad(object):
    def __init__(self, model, box, depth):
        self.model = model
        self.box = box
        self.depth = depth
        hist = self.model.im.crop(self.box).histogram()
        self.color, self.error = color_from_histogram(hist)
        self.leaf = self.is_leaf()
        self.area = self.compute_area()
        self.children = []
    def is_leaf(self):
        l, t, r, b = self.box
        return int(r - l <= LEAF_SIZE or b - t <= LEAF_SIZE)
    def compute_area(self):
        l, t, r, b = self.box
        return (r - l) * (b - t)
    def split(self):
        l, t, r, b = self.box
        lr = l + (r - l) / 2
        tb = t + (b - t) / 2
        depth = self.depth + 1
        tl = Quad(self.model, (l, t, lr, tb), depth)
        tr = Quad(self.model, (lr, t, r, tb), depth)
        bl = Quad(self.model, (l, tb, lr, b), depth)
        br = Quad(self.model, (lr, tb, r, b), depth)
        self.children = (tl, tr, bl, br)
        return self.children
    def get_leaf_nodes(self, max_depth=None):
        if not self.children:
            return [self]
        if max_depth is not None and self.depth >= max_depth:
            return [self]
        result = []
        for child in self.children:
            result.extend(child.get_leaf_nodes(max_depth))
        return result

class Model(object):
    def __init__(self, path):
        self.im = Image.open(path).convert('RGB')
        self.width, self.height = self.im.size
        self.heap = []
        self.root = Quad(self, (0, 0, self.width, self.height), 0)
        self.error_sum = self.root.error * self.root.area
        self.push(self.root)
    @property
    def quads(self):
        return [x[-1] for x in self.heap]
    def average_error(self):
        return self.error_sum / (self.width * self.height)
    def push(self, quad):
        score = -quad.error * (quad.area ** AREA_POWER)
        heapq.heappush(self.heap, (quad.leaf, score, quad))
    def pop(self):
        return heapq.heappop(self.heap)[-1]
    def split(self):
        quad = self.pop()
        self.error_sum -= quad.error * quad.area
        children = quad.split()
        for child in children:
            self.push(child)
            self.error_sum += child.error * child.area
    def render(self, path, max_depth=None):
        m = OUTPUT_SCALE
        dx, dy = (PADDING, PADDING)
        im = Image.new('RGB', (self.width * m + dx, self.height * m + dy))
        draw = ImageDraw.Draw(im)
        draw.rectangle((0, 0, self.width * m, self.height * m), FILL_COLOR)
        for quad in self.root.get_leaf_nodes(max_depth):
            l, t, r, b = quad.box
            box = (l * m + dx, t * m + dy, r * m - 1, b * m - 1)
            if MODE == MODE_ELLIPSE:
                draw.ellipse(box, quad.color)
            elif MODE == MODE_ROUNDED_RECTANGLE:
                radius = m * min((r - l), (b - t)) / 4
                rounded_rectangle(draw, box, radius, quad.color)
            else:
                draw.rectangle(box, quad.color)
        del draw
        if not os.path.exists('result'):os.makedirs('result')
        path=os.path.join(os.path.join(os.getcwd(),'result'),path)
        global iter,ITERATIONS
        if iter['start']!=iter['end'] and iter['start']!='0': im.save(path, 'PNG')
        elif iter['start']=='0' and ITERATIONS==iter['end']: im.save(path, 'PNG')

def main():
    global ITERATIONS
    args = sys.argv[1:]
    if len(args) ==0:
        args.append(raw_input('please input the path of the image:\n'))
        ITERATIONS=raw_input('Please enter the number of times you want to iterate(example:[start:end:step]):\n')
    elif len(args) > 2:
        print 'Usage: python main.py <input_image> [<times_of_iterate>].\nPs:Use python 2.xx to run'
        return
    else:
        ITERATIONS=args[1]
    print '-' * 32
    print '-' * 32
    print '-' * 32
    model = Model(args[0])
    if re.match('^\d+$',ITERATIONS):
        ITERATIONS=int(ITERATIONS)
        iter['start']='0'
        iter['end']=ITERATIONS
        iter['step']=1
    elif re.match('^(\d*):(\d*)$',ITERATIONS):
        m=re.match('^(\d*):(\d*)$',ITERATIONS)
        iter['start']=int(m.groups()[0])
        iter['end']=int(m.groups()[1])
        iter['step']=5
        if iter['start']>iter['end']: return
        print(iter['start'],iter['end'],iter['step'])
    elif re.match('^(\d*):(\d*):(\d*)$',ITERATIONS):
        m=re.match('^(\d*):(\d*):(\d*)$',ITERATIONS)
        iter['start']=int(m.groups()[0])
        iter['end']=int(m.groups()[1])
        iter['step']=int(m.groups()[2])
        if iter['start']>iter['end']: return
        print(iter['start'],iter['end'],iter['step'])
    else:
        print 'error input'
        return

    ITERATIONS=int(iter['start'])

    count=1

    while ITERATIONS <= iter['end'] :
        print '-' * 32
        print 'time',count
        print ' ' * 32
        count+=1

        previous = None
        for i in range(ITERATIONS):
            error = model.average_error()
            if previous is None or previous - error > ERROR_RATE:
                #print i, error
                if SAVE_FRAMES:
                    model.render('frames/%06d.png' % i)
                previous = error
            model.split()
        File_pre_name=os.path.splitext(os.path.basename((args[0])))[0]
        Output_file_name=File_pre_name+'_'+'{:0>6}'.format(str(len(model.quads)))+'_output.png'
        model.render(Output_file_name)
        depth = Counter(x.depth for x in model.quads)
        for key in sorted(depth):
            value = depth[key]
            n = 4 ** key
            pct = 100.0 * value / n
            print '%3d %8d %8d %8.2f%%' % (key, n, value, pct)
        print '-' * 32
        print '             %8d %8.2f%%' % (len(model.quads), 100)
        ITERATIONS+=iter['step']
        if ITERATIONS>iter['end'] and ITERATIONS<(iter['end']+iter['step']) : ITERATIONS=iter['end']
        print '-' * 32
        print ' ' * 32
    result_file=os.path.join('result',Output_file_name)
    if iter['start']=='0':os.startfile(result_file)
    else:
        is_gif=(raw_input('would you want to merge these png to the gif?')).lower()
        os.startfile(os.path.dirname(result_file))
        if is_gif == 'yes':
            os.startfile(os.path.dirname(result_file))
            print 'please wait a moment............................'
            os.chdir(os.path.dirname(result_file))
            images = []
            filenames=sorted((fn for fn in os.listdir('.') if fn.endswith('.png')))
            for filename in filenames:
                images.append(imageio.imread(filename))
            imageio.mimsave(File_pre_name+'_gif.gif', images,duration=0.3)
            os.startfile(File_pre_name+'_gif.gif')
        else:
            os.startfile(os.path.dirname(result_file))

if __name__ == '__main__':
    main()
