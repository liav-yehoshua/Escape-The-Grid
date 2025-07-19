import pygame
import sys
import json
import random
import numpy as np
from queue import PriorityQueue
from sklearn.neural_network import MLPRegressor
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Initialize Pygame
pygame.init()

# ─── Modern UI Fonts ──────────────────────────────────────────────────────────
FONT_REGULAR = pygame.font.SysFont('Segoe UI', 20)
FONT_TITLE   = pygame.font.SysFont('Segoe UI', 48)
FONT_BUTTON  = pygame.font.SysFont('Segoe UI', 36)
FONT_ARROW   = pygame.font.SysFont('Segoe UI', 48)

# ─── Modern Color Palette ─────────────────────────────────────────────────────
COLOR_BG           = (34, 34, 34)
COLOR_GRID_LIGHT   = (60, 60, 60)
COLOR_GRID_DARK    = (50, 50, 50)
COLOR_ACCENT       = (0, 204, 153)
COLOR_BUTTON       = (45, 45, 45)
COLOR_BUTTON_HOVER = (0, 204, 153)
COLOR_TEXT         = (230, 230, 230)
COLOR_LOG          = (200, 155, 100)

# ─── Screen setup ─────────────────────────────────────────────────────────────
WIDTH, HEIGHT = pygame.display.Info().current_w, pygame.display.Info().current_h
GRID_SIZE     = 10
CELL_SIZE     = min(WIDTH // GRID_SIZE, HEIGHT // GRID_SIZE)
screen        = pygame.display.set_mode((WIDTH, HEIGHT), pygame.FULLSCREEN)
pygame.display.set_caption("Escape The Grid")

# ─── Load and scale images ─────────────────────────────────────────────────────
door_image = pygame.transform.scale(
    pygame.image.load(os.path.join(BASE_DIR, 'door.png')), (CELL_SIZE, CELL_SIZE)
)

# Animal image filenames and names
ANIMAL_IMAGES = [
    ("fox.png", "Fox"),
    ("lion.png", "Lion"),
    ("zebra.png", "Zebra"),
    ("penguin.png", "Penguin")
]

player1_images = [
    pygame.transform.scale(
        pygame.image.load(os.path.join(BASE_DIR, fname)), (CELL_SIZE, CELL_SIZE)
    ) for fname, _ in ANIMAL_IMAGES
]
player2_images = list(player1_images)
player1_img, player2_img = player1_images[0], player2_images[1]

# Store animal names for each player
player1_animal = ANIMAL_IMAGES[0][1]
player2_animal = ANIMAL_IMAGES[1][1]

log_image_raw = pygame.image.load(os.path.join(BASE_DIR, 'log.png'))

# ─── A* Pathfinding Functions ─────────────────────────────────────────────────
def heuristic(a, b):
    return abs(b[0]-a[0]) + abs(b[1]-a[1])

def get_neighbors(x, y, logs, can_hop, has_jumped):
    neighbors = []
    for dx, dy in [(0,1),(1,0),(0,-1),(-1,0)]:
        nx, ny = x+dx, y+dy
        if 0 <= nx < GRID_SIZE and 0 <= ny < GRID_SIZE:
            blocked = any(
                log.x<=nx<log.x+(2 if log.horizontal else 1) and
                log.y<=ny<log.y+(2 if not log.horizontal else 1)
                for log in logs
            )
            if not blocked:
                neighbors.append((nx, ny))
            elif can_hop and not has_jumped:
                hx, hy = nx+dx, ny+dy
                if 0<=hx<GRID_SIZE and 0<=hy<GRID_SIZE and not any(
                    log.x<=hx<log.x+(2 if log.horizontal else 1) and
                    log.y<=hy<log.y+(2 if not log.horizontal else 1)
                    for log in logs
                ):
                    neighbors.append((hx, hy))
    return neighbors

def a_star(start, goal, logs, can_hop, has_jumped, opponent_pos=None):
    frontier = PriorityQueue()
    frontier.put((0, start))
    came_from = {start: None}
    cost = {start: 0}
    while not frontier.empty():
        _, current = frontier.get()
        if current == goal:
            path = []
            while current:
                path.append(current)
                current = came_from[current]
            return list(reversed(path))
        for nxt in get_neighbors(current[0], current[1], logs, can_hop, has_jumped):
            new_cost = cost[current] + 1
            # Penalty for being adjacent to opponent
            if opponent_pos:
                if abs(nxt[0] - opponent_pos[0]) <= 1 and abs(nxt[1] - opponent_pos[1]) <= 1:
                    new_cost += 5  # Large penalty for being near opponent
            # Penalty for being adjacent to a log
            for log in logs:
                log_cells = []
                if log.horizontal:
                    log_cells = [(log.x, log.y), (log.x+1, log.y)]
                else:
                    log_cells = [(log.x, log.y), (log.x, log.y+1)]
                for lx, ly in log_cells:
                    if abs(nxt[0] - lx) <= 1 and abs(nxt[1] - ly) <= 1:
                        new_cost += 2  # Penalty for being near a log
                        break
            if nxt not in cost or new_cost < cost[nxt]:
                cost[nxt] = new_cost
                priority = new_cost + heuristic(goal, nxt)
                frontier.put((priority, nxt))
                came_from[nxt] = current
    return None

# ─── Player & Log Classes ─────────────────────────────────────────────────────
class Player:
    def __init__(self, x, y, image, goal):
        self.x,self.y   = x,y
        self.image      = image
        self.goal       = goal
        self.logs       = 5
        self.path       = []
        self.model      = MLPRegressor(hidden_layer_sizes=(50,50), max_iter=1000, random_state=42)
        self.train_data = []
        self.train_model()
        self.logs_placed = 0
        self.moves_left  = 5
        self.has_jumped  = False

    def draw(self):
        screen.blit(self.image, (GRID_TOPLEFT_X + self.x*CELL_SIZE, GRID_TOPLEFT_Y + self.y*CELL_SIZE))

    def move(self, logs=None, opponent=None):
        # Only follow the existing path, do NOT recalculate
        if self.path and len(self.path) > 1 and self.moves_left > 0:
            nx,ny = self.path[1]
            if abs(nx-self.x)+abs(ny-self.y)>1:
                self.has_jumped=True
            self.x,self.y = nx,ny
            self.path = self.path[1:]
            self.moves_left -= 1

    def train_model(self):
        if len(self.train_data)>100:
            X,y = zip(*self.train_data)
            self.model.fit(X,y)
        else:
            X=np.random.rand(1000,7)
            y=np.random.rand(1000)
            self.model.fit(X,y)

    def load_training_data(self,data):
        self.train_data.extend(data)
        self.train_model()

    def decide_action(self, opp, logs):
        md = abs(self.x-self.goal[0])+abs(self.y-self.goal[1])
        od = abs(opp.x-opp.goal[0])+abs(opp.y-opp.goal[1])
        X = np.array([[self.x,self.y,opp.x,opp.y,self.logs,md,od]])
        p = self.model.predict(X)[0]
        if p>0.7 and self.logs>0: action='place_log'
        elif p>0.3: action='hop'
        else: action='move'
        self.train_data.append((X[0],1 if action=='place_log' else (0.5 if action=='hop' else 0)))
        return action

LOG_BIG_SIZE = (int(CELL_SIZE * 2.5), int(CELL_SIZE * 2.5))
log_image_big = pygame.transform.scale(log_image_raw, LOG_BIG_SIZE)

class Log:
    def __init__(self,x,y,hor):
        self.x,self.y,self.horizontal = x,y,hor
    def draw(self):
        offset_x = GRID_TOPLEFT_X + self.x * CELL_SIZE + (CELL_SIZE - LOG_BIG_SIZE[0]) // 2
        offset_y = GRID_TOPLEFT_Y + self.y * CELL_SIZE + (CELL_SIZE - LOG_BIG_SIZE[1]) // 2
        screen.blit(log_image_big, (offset_x, offset_y))

# ─── End Screen ────────────────────────────────────────────────────────────────
def end_screen(winner):
    screen.fill(COLOR_BG)
    if winner == 'Player 1':
        text = f"The {player1_animal} wins!"
    elif winner == 'Player 2':
        text = f"The {player2_animal} wins!"
    else:
        text = "Draw!"
    surf=FONT_TITLE.render(text,True,COLOR_ACCENT)
    rect=surf.get_rect(center=(WIDTH//2,HEIGHT//2-50))
    screen.blit(surf,rect)
    btn=pygame.Rect(WIDTH//2-100,HEIGHT//2+20,200,50)
    while True:
        mx,my=pygame.mouse.get_pos()
        hover=btn.collidepoint(mx,my)
        draw_button("Main Menu",btn,hover)
        pygame.display.flip()
        for e in pygame.event.get():
            if e.type==pygame.QUIT: pygame.quit();sys.exit()
            if e.type==pygame.MOUSEBUTTONDOWN and hover: return

# ─── Initialization ────────────────────────────────────────────────────────────
def initialize_players():
    global player1,player2,logs
    player1=Player(0,0,player1_img,(GRID_SIZE-1,GRID_SIZE-1))
    player2=Player(GRID_SIZE-1,GRID_SIZE-1,player2_img,(0,0))
    logs=[]
    player1.logs_placed=player2.logs_placed=0
    player1.moves_left=player2.moves_left=5
    player1.has_jumped=player2.has_jumped=False

# ─── Drawing Utilities ─────────────────────────────────────────────────────────
# Calculate grid topleft offset for centering
INFO_PANEL_WIDTH = 320
INFO_PANEL_HEIGHT = 200
INFO_PANEL_Y = (HEIGHT - INFO_PANEL_HEIGHT) // 2
INFO_PANEL_LEFT_X = 60

# Calculate grid position first
GRID_PIXEL_SIZE = GRID_SIZE * CELL_SIZE
GRID_TOPLEFT_X = INFO_PANEL_LEFT_X + INFO_PANEL_WIDTH + 60
GRID_TOPLEFT_Y = (HEIGHT - GRID_PIXEL_SIZE) // 2

# Now you can safely use GRID_TOPLEFT_X
INFO_PANEL_RIGHT_X = GRID_TOPLEFT_X + GRID_PIXEL_SIZE + 60

def draw_grid():
    shadow_offset = 6  # pixels to offset the shadow
    for x in range(GRID_SIZE):
        for y in range(GRID_SIZE):
            # Shadow rectangle (drawn first, slightly offset)
            shadow_rect = pygame.Rect(
                GRID_TOPLEFT_X + x*CELL_SIZE + shadow_offset,
                GRID_TOPLEFT_Y + y*CELL_SIZE + shadow_offset,
                CELL_SIZE,
                CELL_SIZE
            )
            pygame.draw.rect(screen, (20, 20, 20), shadow_rect, border_radius=12)

            # Main cell rectangle (drawn on top)
            cell_rect = pygame.Rect(GRID_TOPLEFT_X + x*CELL_SIZE, GRID_TOPLEFT_Y + y*CELL_SIZE, CELL_SIZE, CELL_SIZE)
            c = COLOR_GRID_LIGHT if (x+y)%2==0 else COLOR_GRID_DARK
            pygame.draw.rect(screen, c, cell_rect, border_radius=12)

            # Optional: Add a simple vertical gradient for extra depth
            gradient_surface = pygame.Surface((CELL_SIZE, CELL_SIZE), pygame.SRCALPHA)
            for i in range(CELL_SIZE):
                alpha = int(40 * (1 - i / CELL_SIZE))  # fade out
                pygame.draw.line(gradient_surface, (255,255,255,alpha), (0,i), (CELL_SIZE,i))
            screen.blit(gradient_surface, (GRID_TOPLEFT_X + x*CELL_SIZE, GRID_TOPLEFT_Y + y*CELL_SIZE))
    # Accent border
    pygame.draw.rect(screen, COLOR_ACCENT, pygame.Rect(GRID_TOPLEFT_X, GRID_TOPLEFT_Y, GRID_PIXEL_SIZE, GRID_PIXEL_SIZE), width=4, border_radius=16)

def draw_button(text,rect,hovered=False):
    col = COLOR_BUTTON_HOVER if hovered else COLOR_BUTTON
    pygame.draw.rect(screen,col,rect,border_radius=8)
    txt=FONT_BUTTON.render(text,True,COLOR_TEXT)
    screen.blit(txt,txt.get_rect(center=rect.center))

def draw_arrows(rect):
    l=FONT_ARROW.render("<",True,COLOR_ACCENT)
    r=FONT_ARROW.render(">",True,COLOR_ACCENT)
    lr=l.get_rect(midright=(rect.left-10,rect.centery))
    rr=r.get_rect(midleft=(rect.right+10,rect.centery))
    screen.blit(l,lr);screen.blit(r,rr)
    return lr,rr

def draw_text(txt,pos,font):
    surf=font.render(txt,True,COLOR_TEXT)
    screen.blit(surf,surf.get_rect(topleft=pos))

# ─── Main Draw Function ─────────────────────────────────────────────────────────
def draw(mode):
    screen.fill(COLOR_BG)
    draw_grid()
    for log in logs: log.draw()
    screen.blit(door_image,(GRID_TOPLEFT_X, GRID_TOPLEFT_Y))
    screen.blit(door_image,(GRID_TOPLEFT_X + (GRID_SIZE-1)*CELL_SIZE, GRID_TOPLEFT_Y + (GRID_SIZE-1)*CELL_SIZE))
    player1.draw(); player2.draw()
    # Stylish A* path lines with distinct colors
    for pl, color, glow in [
        (player1, (0, 120, 255), (100, 180, 255, 90)),  # Blue for Player 1
        (player2, (220, 40, 40), (255, 120, 120, 90))   # Red for Player 2
    ]:
        if pl.path:
            # Draw the next 5 steps of the path
            for i in range(len(pl.path)-1):
                sx,sy=pl.path[i]; ex,ey=pl.path[i+1]
                sp=(GRID_TOPLEFT_X + sx*CELL_SIZE+CELL_SIZE//2, GRID_TOPLEFT_Y + sy*CELL_SIZE+CELL_SIZE//2)
                ep=(GRID_TOPLEFT_X + ex*CELL_SIZE+CELL_SIZE//2, GRID_TOPLEFT_Y + ey*CELL_SIZE+CELL_SIZE//2)
                # Glow effect (thicker, semi-transparent line)
                glow_surface = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
                pygame.draw.line(glow_surface, glow, sp, ep, 10)
                screen.blit(glow_surface, (0,0))
                # Main colored line
                pygame.draw.line(screen, color, sp, ep, 4)
    # Info panels
    if mode == 'place_log':
        # Log phase panel (left)
        panel_rect = pygame.Rect(INFO_PANEL_LEFT_X, INFO_PANEL_Y, INFO_PANEL_WIDTH, INFO_PANEL_HEIGHT)
        pygame.draw.rect(screen, COLOR_BUTTON, panel_rect, border_radius=16)
        y = INFO_PANEL_Y + 18
        phase_text = "Log Phase"
        phase_surf = FONT_BUTTON.render(phase_text, True, COLOR_ACCENT)
        phase_rect = phase_surf.get_rect(midtop=(INFO_PANEL_LEFT_X + INFO_PANEL_WIDTH//2, y))
        screen.blit(phase_surf, phase_rect)
        y += phase_rect.height + 18
        draw_text(f"P1 Logs: {player1.logs}", (INFO_PANEL_LEFT_X + 30, y), FONT_REGULAR)
        y += 40
        draw_text(f"P2 Logs: {player2.logs}", (INFO_PANEL_LEFT_X + 30, y), FONT_REGULAR)
    elif mode == 'move':
        # Move phase panel (right)
        panel_rect = pygame.Rect(INFO_PANEL_RIGHT_X, INFO_PANEL_Y, INFO_PANEL_WIDTH, INFO_PANEL_HEIGHT)
        pygame.draw.rect(screen, COLOR_BUTTON, panel_rect, border_radius=16)
        y = INFO_PANEL_Y + 18
        phase_text = "Move Phase"
        phase_surf = FONT_BUTTON.render(phase_text, True, COLOR_ACCENT)
        phase_rect = phase_surf.get_rect(midtop=(INFO_PANEL_RIGHT_X + INFO_PANEL_WIDTH//2, y))
        screen.blit(phase_surf, phase_rect)
        y += phase_rect.height + 18
        draw_text(f"P1 Moves: {player1.moves_left}", (INFO_PANEL_RIGHT_X + 30, y), FONT_REGULAR)
        y += 40
        draw_text(f"P2 Moves: {player2.moves_left}", (INFO_PANEL_RIGHT_X + 30, y), FONT_REGULAR)
    # Back button in bottom left corner
    back=pygame.Rect(30, HEIGHT-70, 140, 40)
    draw_button("Back", back, back.collidepoint(pygame.mouse.get_pos()))
    pygame.display.flip()
    return back

# ─── Logs & Turns ───────────────────────────────────────────────────────────────
def logs_overlap(nl,elogs):
    for log in elogs:
        if nl.horizontal==log.horizontal:
            if nl.horizontal and nl.y==log.y and nl.x<log.x+2 and log.x<nl.x+2: return True
            if not nl.horizontal and nl.x==log.x and nl.y<log.y+2 and log.y<nl.y+2: return True
        else:
            if nl.horizontal and ((log.x<=nl.x<log.x+1 or log.x<=nl.x+1<log.x+1) and (nl.y<=log.y<nl.y+1 or nl.y<=log.y+1<nl.y+1)): return True
            if not nl.horizontal and ((nl.x<=log.x<nl.x+1 or nl.x<=log.x+1<nl.x+1) and (log.y<=nl.y<log.y+1 or log.y<=nl.y+1<log.y+1)): return True
    return False

def place_log(p,o):
    if p.logs>0:
        for _ in range(50):
            horiz=random.choice([True,False])
            x=random.randint(0,GRID_SIZE-2) if horiz else random.randint(0,GRID_SIZE-1)
            y=random.randint(0,GRID_SIZE-1) if horiz else random.randint(0,GRID_SIZE-2)
            nl=Log(x,y,horiz)
            if not logs_overlap(nl,logs) and (x,y) not in [(p.x,p.y),(o.x,o.y),(0,0),(GRID_SIZE-1,GRID_SIZE-1)]:
                tmp=logs+[nl]
                if a_star((p.x,p.y),p.goal,tmp,True,False) and a_star((o.x,o.y),o.goal,tmp,True,False):
                    logs.append(nl); p.logs-=1; return True
    return False

def player_turn(p,o,mode):
    if mode=='place_log' and p.logs_placed<5 and place_log(p,o): p.logs_placed+=1
    elif mode=='move' and p.moves_left>0:
        p.move(logs=logs, opponent=o)
        # If path is blocked, player does not move

def check_win(p): return (p.x,p.y)==p.goal
def clear_logs(): global logs; logs=[]

def play_game():
    turn,mode,winner=0,'place_log',None
    clock=pygame.time.Clock(); exit_menu=False
    while turn<1000 and not exit_menu and winner is None:
        back=draw(mode)
        for e in pygame.event.get():
            if e.type==pygame.QUIT: pygame.quit();sys.exit()
            if e.type==pygame.MOUSEBUTTONDOWN and back.collidepoint(e.pos): exit_menu=True
        if exit_menu: break
        if mode=='place_log' and player1.logs_placed==5 and player2.logs_placed==5:
            mode='move'
            player1.path = a_star((player1.x, player1.y), player1.goal, logs, True, player1.has_jumped, (player2.x, player2.y))
            player2.path = a_star((player2.x, player2.y), player2.goal, logs, True, player2.has_jumped, (player1.x, player1.y))
            player1.moves_left=player2.moves_left=5
            player1.has_jumped=player2.has_jumped=False
        if turn%2==0: player_turn(player1,player2,mode)
        else:        player_turn(player2,player1,mode)
        if mode=='move':
            if check_win(player1): winner='Player 1'
            if check_win(player2): winner='Player 2'
            if player1.moves_left==0 and player2.moves_left==0:
                mode='place_log'; clear_logs()
                player1.logs=player2.logs=5; player1.logs_placed=player2.logs_placed=0
        turn+=1; clock.tick(2)
    if exit_menu: return 'menu'
    end_screen(winner)
    return winner

def save_game_data(p1,p2,lg,win):
    gd={'player1':{'x':p1.x,'y':p1.y,'logs':p1.logs},
        'player2':{'x':p2.x,'y':p2.y,'logs':p2.logs},
        'logs':[{'x':l.x,'y':l.y,'horizontal':l.horizontal} for l in lg],
        'winner':win,'training_data':{'player1':p1.train_data,'player2':p2.train_data}}
    try: data=json.load(open('game_data.json'))
    except: data=[]
    data.append(gd)
    json.dump(data,open('game_data.json','w'),indent=4)

def load_training_data():
    try: data=json.load(open('game_data.json'))
    except: data=[]
    p1=[]; p2=[]
    for g in data: p1.extend(g['training_data']['player1']); p2.extend(g['training_data']['player2'])
    return p1,p2

def choose_player_images():
    global player1_img, player2_img, player1_animal, player2_animal
    chosen1,chosen2=False,False
    while not(chosen1 and chosen2):
        screen.fill(COLOR_BG)
        for i,img in enumerate(player1_images):
            screen.blit(img,(WIDTH//4-CELL_SIZE//2,HEIGHT//4+i*(CELL_SIZE+10)))
        for i,img in enumerate(player2_images):
            screen.blit(img,(WIDTH*3//4-CELL_SIZE//2,HEIGHT//4+i*(CELL_SIZE+10)))
        title1=FONT_TITLE.render("Choose Player 1",True,COLOR_TEXT)
        screen.blit(title1,title1.get_rect(center=(WIDTH//4,HEIGHT//4-40)))
        title2=FONT_TITLE.render("Choose Player 2",True,COLOR_TEXT)
        screen.blit(title2,title2.get_rect(center=(WIDTH*3//4,HEIGHT//4-40)))
        pygame.display.flip()
        for e in pygame.event.get():
            if e.type==pygame.QUIT:pygame.quit();sys.exit()
            if e.type==pygame.MOUSEBUTTONDOWN:
                mx,my=e.pos
                for i,img in enumerate(player1_images):
                    r1=pygame.Rect(WIDTH//4-CELL_SIZE//2,HEIGHT//4+i*(CELL_SIZE+10),CELL_SIZE,CELL_SIZE)
                    if r1.collidepoint(mx,my):
                        player1_img,player1_animal=img,ANIMAL_IMAGES[i][1];chosen1=True
                for i,img in enumerate(player2_images):
                    r2=pygame.Rect(WIDTH*3//4-CELL_SIZE//2,HEIGHT//4+i*(CELL_SIZE+10),CELL_SIZE,CELL_SIZE)
                    if r2.collidepoint(mx,my):
                        player2_img,player2_animal=img,ANIMAL_IMAGES[i][1];chosen2=True

def main_menu():
    sr=pygame.Rect(WIDTH//2-150,HEIGHT//2+60,300,50)
    cr=pygame.Rect(WIDTH//2-150,HEIGHT//2+130,300,50)
    er=pygame.Rect(WIDTH//2-150,HEIGHT//2+200,300,50)
    ng=1
    while True:
        screen.fill(COLOR_BG)
        title_surf=FONT_TITLE.render("Escape The Grid",True,COLOR_TEXT)
        screen.blit(title_surf,title_surf.get_rect(center=(WIDTH//2,HEIGHT//2-220)))
        # Modern Number of Games selector (higher up)
        label_surf = FONT_BUTTON.render("Number of Games", True, COLOR_ACCENT)
        label_rect = label_surf.get_rect(center=(WIDTH//2, HEIGHT//2-120))
        screen.blit(label_surf, label_rect)
        # Large number with glowing pill and shadow
        num_center = (WIDTH//2, HEIGHT//2-20)
        pill_width, pill_height = 120, 90
        pill_rect = pygame.Rect(0,0,pill_width,pill_height)
        pill_rect.center = num_center
        # Glow effect
        glow_surf = pygame.Surface((pill_width+24, pill_height+24), pygame.SRCALPHA)
        pygame.draw.ellipse(glow_surf, (*COLOR_ACCENT, 60), glow_surf.get_rect())
        screen.blit(glow_surf, (pill_rect.x-12, pill_rect.y-12))
        # Pill background
        pygame.draw.ellipse(screen, COLOR_BUTTON, pill_rect)
        pygame.draw.ellipse(screen, COLOR_ACCENT, pill_rect, 4)
        # Number shadow
        num_surf_shadow = FONT_TITLE.render(f"{ng}", True, (0,0,0))
        num_rect = num_surf_shadow.get_rect(center=(num_center[0], num_center[1]+4))
        screen.blit(num_surf_shadow, num_rect)
        # Number
        num_surf = FONT_TITLE.render(f"{ng}", True, COLOR_TEXT)
        num_rect = num_surf.get_rect(center=num_center)
        screen.blit(num_surf, num_rect)
        # Large arrow buttons
        arrow_size = 60
        left_arrow_rect = pygame.Rect(num_rect.left-arrow_size-30, num_rect.centery-arrow_size//2, arrow_size, arrow_size)
        right_arrow_rect = pygame.Rect(num_rect.right+30, num_rect.centery-arrow_size//2, arrow_size, arrow_size)
        # Draw arrows with hover effect
        mouse_pos = pygame.mouse.get_pos()
        for rect, symbol in [(left_arrow_rect, "<"), (right_arrow_rect, ">")]:
            hovered = rect.collidepoint(mouse_pos)
            col = COLOR_BUTTON_HOVER if hovered else COLOR_BUTTON
            pygame.draw.rect(screen, col, rect, border_radius=16)
            arrow_surf = FONT_TITLE.render(symbol, True, COLOR_ACCENT)
            arrow_rect = arrow_surf.get_rect(center=rect.center)
            screen.blit(arrow_surf, arrow_rect)
        # Main menu buttons (lower)
        draw_button("Start Game",sr,sr.collidepoint(pygame.mouse.get_pos()))
        draw_button("Choose Players",cr,cr.collidepoint(pygame.mouse.get_pos()))
        draw_button("Exit",er,er.collidepoint(pygame.mouse.get_pos()))
        pygame.display.flip()
        for e in pygame.event.get():
            if e.type==pygame.QUIT:return 'exit',0
            if e.type==pygame.MOUSEBUTTONDOWN:
                if sr.collidepoint(e.pos):return 'start',ng
                if cr.collidepoint(e.pos):choose_player_images()
                if er.collidepoint(e.pos):return 'exit',0
                if left_arrow_rect.collidepoint(e.pos):ng=max(1,ng-1)
                if right_arrow_rect.collidepoint(e.pos):ng+=1
            if e.type==pygame.KEYDOWN:
                if e.key==pygame.K_UP:ng+=1
                if e.key==pygame.K_DOWN:ng=max(1,ng-1)

def main():
    p1_data,p2_data=load_training_data()
    while True:
        action,ng=main_menu()
        if action=='exit':pygame.quit();sys.exit()
        w1=w2=0
        for _ in range(ng):
            initialize_players()
            player1.load_training_data(p1_data)
            player2.load_training_data(p2_data)
            win=play_game()
            if win=='Player 1':w1+=1
            if win=='Player 2':w2+=1
            player1.train_model();player2.train_model()
        print(f"Final Score - P1: {w1}, P2: {w2}")

if __name__ == '__main__':
    main()