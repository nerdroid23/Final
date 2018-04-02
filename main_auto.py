x = 8
y = 30

import os
import threading as t

os.environ['SDL_VIDEO_WINDOW_POS'] = "%d,%d" % (x, y)
import pygame
pygame.init()
from pygame.locals import *
import math
import pdb
import car
from argparse import ArgumentParser
import pickle

parser = ArgumentParser()
parser.add_argument("filename", help="name of trained model file")
modelname = parser.parse_args()

filename = 'models/' + modelname.filename
model = pickle.load(open(filename, 'rb'))

slow = False

# import movie
# screen_size = (1200,900)
screen_size = (1024, 768)
screen = pygame.display.set_mode(screen_size)
MAX_LAPS = 10
car.MAX_LAPS = MAX_LAPS
screen.fill((0, 192, 0))
clock = pygame.time.Clock()
running = True
red = car.Sprite()
red.Load('red', 360)
car.xs = screen_size[0] / 2;
car.ys = screen_size[1] / 2;
track = pygame.image.load('track.png')
visible_track = pygame.image.load('track_textured2.png')
trap = pygame.Rect(844, 1324, 140, 200)
trk = track.get_at((0, 0))

finish = pygame.Rect(600,250,90,130)

# track.set_colorkey(trk)
inbox = trap.collidepoint(red.xc, red.yc)
lap = 0
# pdb.set_trace()
frames = 0
while running:
    clock.tick(30)
    frames = frames + 1
    car.frames = frames
    screen.fill((0, 192, 0))
    red.Update()
    if trap.collidepoint(red.xc, red.yc) == 0:
        if inbox == 1:
            red.lap += 1
            inbox = 0
    else:
        inbox = 1
    screen.blit(visible_track, (car.xs - red.xc, car.ys - red.yc))

    if finish.collidepoint(red.xc,red.yc) == 1:
        for i in range(100):
            clock.tick(24)
        exit()

    predicted = model.predict([[red.xc, red.yc]])
    d = predicted

    red.Draw(car.xs, car.ys, screen)
    # movie.make_movie(screen,frames)
    pygame.display.flip()
    onboard = True
    if red.xc >= track.get_width():
        onboard = False
    if red.xc < 0:
        onboard = False
    if red.yc >= track.get_height():
        onboard = False
    if red.yc < 0:
        onboard = False
    if onboard:
        clr = track.get_at((red.xc, red.yc))
    else:
        clr = trk
    # pdb.set_trace()
    red.wobble = 0
    if red.lap > MAX_LAPS:
        # print 'Total Time = ',frames/24

        for i in range(300):
            clock.tick(24)
        exit()
    if clr == trk:
        # print 'Off track'
        # red.wobble = 10
        # pdb.set_trace()
        if red.gear > 1:
            red.gear = 1
            red.speed = 1

    # Tactic
    key = pygame.key.get_pressed()
    if key[K_s]:
        slow = True
    if slow:
        red.speed = .95 * red.speed - .05 * (2.5 * red.gear)
    
    if red.gear > 0:
        if d == 1:
            red.view = (red.view + 2) % 360
        elif d == 2:
            red.view = (red.view + 358) % 360

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == KEYDOWN:
            if event.key == K_ESCAPE:
                running = False;
            elif event.key == K_UP:
                red.gear = red.gear + 1
                if red.gear < 4:
                    red.Shift_Up()
                if red.gear > 4:
                    red.gear = 4
            elif event.key == K_DOWN:
                red.gear = red.gear - 1
                if red.gear < 0:
                    red.gear = 0

    print str(red.xc) + " " + str(red.yc) + " " + str(d)
