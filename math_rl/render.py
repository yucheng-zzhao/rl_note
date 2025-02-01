from typing import Union

import matplotlib.animation as animation
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np

np.random.seed(1)
class Render:
    def __init__(self, target: Union[list, tuple, np.ndarray], forbidden: Union[list, tuple, np.ndarray],
                 size=5):
        self.agent = None
        self.target = target
        self.forbidden = forbidden
        self.size = size
        self.fig = plt.figure(figsize=(10, 10), dpi=self.size * 20)
        self.ax = plt.gca()
        self.ax.xaxis.set_ticks_position('top')
        self.ax.invert_yaxis()
        self.ax.xaxis.set_ticks(range(0, size + 1))
        self.ax.yaxis.set_ticks(range(0, size + 1))
        self.ax.grid(True, linestyle="-", color="gray", linewidth="1", axis='both')
        self.ax.tick_params(bottom=False, left=False, right=False, top=False, labelbottom=False, labelleft=False,
                            labeltop=False)

        for y in range(size):
            self.write_word(pos=(-0.6, y), word=str(y), size_discount=0.8)
            self.write_word(pos=(y, -0.6), word=str(y), size_discount=0.8)

        for pos in self.forbidden:
            self.fill_block(pos=pos)
        self.fill_block(pos=self.target, color='darkturquoise')
        self.trajectory = []
        self.agent = patches.Arrow(-10, -10, 0.4, 0, color='red', width=0.5)
        self.ax.add_patch(self.agent)

    def fill_block(self, pos: Union[list, tuple, np.ndarray], color: str = '#EDB120', width=1.0,
                   height=1.0) -> patches.RegularPolygon:
        return self.ax.add_patch(
            patches.Rectangle((pos[0], pos[1]),
                              width=1.0,
                              height=1.0,
                              facecolor=color,
                              fill=True,
                              alpha=0.90,
                              ))

    def draw_random_line(self, pos1: Union[list, tuple, np.ndarray], pos2: Union[list, tuple, np.ndarray]) -> None:
        offset1 = np.random.uniform(low=-0.1, high=0.1, size=1)
        offset2 = np.random.uniform(low=-0.1, high=0.1, size=1)
        x = [pos1[0] + 0.5, pos2[0] + 0.5]
        y = [pos1[1] + 0.5, pos2[1] + 0.5]
        if pos1[0] == pos2[0]:
            x = [x[0] + offset1, x[1] + offset2]
        else:
            y = [y[0] + offset1, y[1] + offset2]
        self.ax.plot(x, y, color='g', scalex=False, scaley=False)


    def draw_circle(self, pos: Union[list, tuple, np.ndarray], radius: float,
                    color: str = 'green', fill: bool = True) -> patches.CirclePolygon:
        return self.ax.add_patch(
            patches.Circle((pos[0] + 0.5, pos[1] + 0.5),
                           radius=radius,
                           facecolor=color,
                           edgecolor='green',
                           linewidth=2,
                           fill=fill
                           ))

    def draw_action(self, pos: Union[list, tuple, np.ndarray], toward: Union[list, tuple, np.ndarray],
                    color: str = 'green', radius: float = 0.10) -> None:
        if not np.array_equal(np.array(toward), np.array([0, 0])):
            self.ax.add_patch(
                patches.Arrow(pos[0] + 0.5, pos[1] + 0.5, dx=toward[0],
                              dy=toward[1], color=color, width=0.05 + 0.05 * np.linalg.norm(np.array(toward) / 0.5),
                              linewidth=0.5))
        else:
            self.draw_circle(pos=tuple(pos), color='white', radius=radius, fill=False)
            

    def write_word(self, pos: Union[list, np.ndarray, tuple], word: str, color: str = 'black', y_offset: float = 0,
                   size_discount: float = 1.0) -> None:
        self.ax.text(pos[0] + 0.5, pos[1] + 0.5 + y_offset, word, size=size_discount * (30 - 2 * self.size), ha='center',
                 va='center', color=color)

    def upgrade_agent(self, pos: Union[list, np.ndarray, tuple], action,
                      next_pos: Union[list, np.ndarray, tuple], ) -> None:
        self.trajectory.append([tuple(pos), action, tuple(next_pos)])

    def show_frame(self, t: float = 0.2) -> None:
        self.fig.show()
        

    def save_frame(self, name: str) -> None:
        self.fig.savefig(name)

    def save_video(self, name: str) -> None:
        anim = animation.FuncAnimation(self.fig, self.animate, init_func=self.init(), frames=len(self.trajectory),
                                       interval=25, repeat=False)
        anim.save(name + '.mp4')


    def animate(self, i):
        print(i,len(self.trajectory))
        location = self.trajectory[i][0]
        action = self.trajectory[i][1]
        next_location = self.trajectory[i][2]
        next_location = np.clip(next_location, -0.4, self.size - 0.6)
        self.agent.remove()
        if action[0] + action[1] != 0:
            self.agent = patches.Arrow(x=location[0] + 0.5, y=location[1] + 0.5,
                                       dx=action[0] / 2, dy=action[1] / 2,
                                       color='b',
                                       width=0.5)
        else:
            self.agent = patches.Circle(xy=(location[0] + 0.5, location[1] + 0.5),
                                        radius=0.15, fill=True, color='b',
                                        )
        self.ax.add_patch(self.agent)

        self.draw_random_line(pos1=location, pos2=next_location)

    def draw_episode(self):
        for i in range(len(self.trajectory)):
            location = self.trajectory[i][0]
            next_location = self.trajectory[i][2]
            self.draw_random_line(pos1=location, pos2=next_location)

    def add_subplot_to_fig(self, fig, x, y, subplot_position, xlabel, ylabel, title=''):
        ax = fig.add_subplot(subplot_position)
        ax.plot(x, y)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)


if __name__ == '__main__':
    render = Render(target=[4, 4], forbidden=[np.array([1, 2]), np.array([2, 2])], size=5)
    render.draw_action(pos=[3, 3], toward=(0, 0.4))

    for num in range(100):
        render.draw_random_line(pos1=[1.5, 1.5], pos2=[1.5, 2.5])

    action_to_direction = {
        0: np.array([-1, 0]),
        1: np.array([0, 1]),
        2: np.array([1, 0]),
        3: np.array([0, -1]),
        4: np.array([0, 0]),
    }
    uniform_policy = np.random.random(size=(25, 5))
    for a in range(4):
        render.trajectory.append([(a, a), None, (a+1, a+1)])
    render.draw_episode()
    render.save_frame('test.jpg')