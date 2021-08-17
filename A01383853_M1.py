!pip3 install mesa

# La clase `Model` se hace cargo de los atributos a nivel del modelo, maneja los agentes. 
# Cada modelo puede contener múltiples agentes y todos ellos son instancias de la clase `Agent`.
from mesa import Agent, Model 

# Debido a que necesitamos un solo agente por celda elegimos `SingleGrid` que fuerza un solo objeto por celda.
from mesa.space import SingleGrid

# Con `SimultaneousActivation` hacemos que todos los agentes se activen de manera simultanea.
from mesa.time import SimultaneousActivation

# Vamos a hacer uso de `DataCollector` para obtener el grid completo cada paso (o generación) y lo usaremos para graficarlo.
from mesa.datacollection import DataCollector

# mathplotlib lo usamos para graficar/visualizar como evoluciona el autómata celular.
%matplotlib inline
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
plt.rcParams["animation.html"] = "jshtml"
matplotlib.rcParams['animation.embed_limit'] = 2**128

# Definimos los siguientes paquetes para manejar valores númericos.
import numpy as np
import pandas as pd

# Definimos otros paquetes que vamos a usar para medir el tiempo de ejecución de nuestro algoritmo.
import time
import datetime

def get_grid(model):
  grid=np.zeros((model.grid.width,model.grid.height))
  '''
  for cell in model.grid.coord_iter():
    cell_content, x, y=cell
    grid[x][y]=cell_content.dirty
  '''
  for cell in model.grid.coord_iter():
    cell_content, x, y=cell
    for content in cell_content:
      if isinstance(contenido,RobotCleaner):
        grid[x][y]=2
      else:
        grid[x][y]=cell_content.state
  
  return grid

class RobotCleaner(Agent):
  def __init__(self,unique_id,model):
    super().__init__(unique_id,model)
    self.live = np.random.choice([0,1])
    self.next_state=None
  
  def step(self):
    dirty_neighbours=0

    neighbours=self.model.grid.get_neighbors(
        self.pos,
        moore=True,
        include_center=False)
    
    for neighbor in neighbours:
      dirty_neighbours=dirty_neighbours+neighbor.dirty
    
    self.next_state=self.dirty
    if self.next_state == 1:
        if live_neighbours < 2 or live_neighbours > 3:
            self.next_state = 0
    else:
        if live_neighbours == 3:
            self.next_state = 1

  def advance(self):
    self.dirty=self.next_state


class RobotCleanerModel(Model):
  def __init__(self,width,height,agents):
    self.num_agents=agents
    self.grid=SingleGrid(width,height,False)
    self.schedule=SimultaneousActivation(self)

    for (content,x,y) in self.grid.coord_iter():
      a=RobotCleaner((x,y),self)
      self.grid.place_agent(a,(x,y))
      self.schedule.add(a)
    
    self.datacollector=DataCollector(
        model_reporters={"Grid":get_grid})

  def step(self):
    self.datacollector.collect(self)
    self.schedule.step()
    
    GRID_SIZE=10
AgentsQTY=5
t=10

start_time=time.time()
model=RobotCleanerModel(GRID_SIZE,GRID_SIZE,AgentsQTY)

while t:
        mins, secs = divmod(t, 60)
        timer = '{:02d}:{:02d}'.format(mins, secs)
        print(timer, end="\r")
        time.sleep(1)
        '''model.step()'''
        t -= 1

print('Tiempo de ejecucion: ',str(datetime.timedelta(seconds=(time.time()-start_time))))

all_grid = model.datacollector.get_model_vars_dataframe()
print(all_grid)

fig, axs = plt.subplots(figsize=(7,7))
axs.set_xticks([])
axs.set_yticks([])
patch = plt.imshow(all_grid.iloc[0][0], cmap=plt.cm.binary)

def animate(i):
    patch.set_data(all_grid.iloc[i][0])
    
anim = animation.FuncAnimation(fig, animate, frames=NUM_Agents)
