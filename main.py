import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *
import cv2
import mediapipe as mp
import numpy as np
import random

mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_drawing = mp.solutions.drawing_utils

pygame.init()
screen = pygame.display.set_mode((980, 720), DOUBLEBUF | OPENGL)
pygame.display.set_caption("magic cube")
clock = pygame.time.Clock()

cap = cv2.VideoCapture(0)

hand_pos = [0.0, 0.0, -5.0]
particles = []
cube_rotation = [0.0, 0.0, 0.0]
rotate_cube = False

def draw_translucent_cube():
    vertices = [
        [1, 1, -1], [1, -1, -1], [-1, -1, -1], [-1, 1, -1],
        [1, 1, 1], [1, -1, 1], [-1, -1, 1], [-1, 1, 1]
    ]
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),
        (4, 5), (5, 6), (6, 7), (7, 4),
        (0, 4), (1, 5), (2, 6), (3, 7)
    ]
    glColor4f(0.5, 0.0, 0.7, 0.3)
    glBegin(GL_QUADS)
    for edge in edges:
        for vertex in edge:
            glVertex3fv(vertices[vertex])
    glEnd()
    glColor4f(1, 1, 1, 1)
    glBegin(GL_LINES)
    for edge in edges:
        for vertex in edge:
            glVertex3fv(vertices[vertex])
    glEnd()

def draw_particles():
    global particles
    color = [random.random(), random.random(), random.random(), 1.0]
    particles.append([hand_pos.copy(), color])
    if len(particles) > 100:
        particles.pop(0)
    glBegin(GL_POINTS)
    for particle in particles:
        position, color = particle
        glColor4f(*color)
        glVertex3f(position[0], position[1], position[2])
    glEnd()

def is_hand_closed(hand_landmarks):
    thumb_tip = hand_landmarks.landmark[4].x
    thumb_ip = hand_landmarks.landmark[3].x
    index_tip = hand_landmarks.landmark[8].y
    index_pip = hand_landmarks.landmark[6].y
    middle_tip = hand_landmarks.landmark[12].y
    middle_pip = hand_landmarks.landmark[10].y
    ring_tip = hand_landmarks.landmark[16].y
    ring_pip = hand_landmarks.landmark[14].y
    pinky_tip = hand_landmarks.landmark[20].y
    pinky_pip = hand_landmarks.landmark[18].y
    if (thumb_tip < thumb_ip and index_tip > index_pip and
        middle_tip > middle_pip and ring_tip > ring_pip and pinky_tip > pinky_pip):
        return True
    return False

running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    ret, frame = cap.read()
    if not ret:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(frame_rgb)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            center_x = hand_landmarks.landmark[9].x
            center_y = hand_landmarks.landmark[9].y
            hand_pos[0] = (center_x - 0.5) * 8
            hand_pos[1] = -(center_y - 0.5) * 8
            if is_hand_closed(hand_landmarks):
                rotate_cube = True
            else:
                rotate_cube = False

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    glLoadIdentity()
    gluPerspective(45, (800/600), 0.1, 50.0)
    glTranslatef(0.0, 0.0, -5)
    glTranslatef(hand_pos[0], hand_pos[1], hand_pos[2])

    if rotate_cube:
        cube_rotation[0] += 1
        cube_rotation[1] += 1
        cube_rotation[2] += 1

    glRotatef(cube_rotation[0], 1, 0, 0)
    glRotatef(cube_rotation[1], 0, 1, 0)
    glRotatef(cube_rotation[2], 0, 0, 1)

    draw_translucent_cube()
    draw_particles()

    pygame.display.flip()
    clock.tick(30)

cap.release()
pygame.quit()
