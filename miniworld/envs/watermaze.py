import numpy as np
import math
import gymnasium as gym
from gymnasium import spaces, utils

from miniworld.entity import Box
from miniworld.miniworld import MiniWorldEnv, Room, gen_texcs_wall, gen_texcs_floor
from miniworld.opengl import Texture
from miniworld.params import DEFAULT_PARAMS
from pyglet.gl import (
    glBegin,
    glEnd,
    glVertex3f,
    glNormal3f,
    glTexCoord2f,
    glColor3f,
    GL_QUADS,
    GL_POLYGON,
)

# Define the vertical vector for convenience
Y_VEC = np.array([0, 1, 0])


class FourColoredRoom(Room):
    def __init__(self, wall_textures, **kwargs):
        super().__init__(**kwargs)
        self.wall_texture_names = wall_textures
        # Ensure we have exactly one texture per wall
        assert len(self.wall_texture_names) == self.num_walls

    def _gen_static_data(self, params, rng):
        # Load textures
        self.wall_textures = [
            Texture.get(name, rng) for name in self.wall_texture_names
        ]
        self.floor_tex = Texture.get(self.floor_tex_name, rng)
        self.ceil_tex = Texture.get(self.ceil_tex_name, rng)

        # Generate the floor vertices
        self.floor_verts = self.outline
        self.floor_texcs = gen_texcs_floor(self.floor_tex, self.floor_verts)

        # Generate the ceiling vertices
        # Flip the ceiling vertex order because of backface culling
        self.ceil_verts = np.flip(self.outline, axis=0) + self.wall_height * Y_VEC
        self.ceil_texcs = gen_texcs_floor(self.ceil_tex, self.ceil_verts)

        self.wall_segs = []
        self.wall_render_data = []  # List of dicts: {tex, verts, norms, texcs}

        for wall_idx in range(self.num_walls):
            tex = self.wall_textures[wall_idx]

            # Temporary lists for this wall
            current_wall_verts = []
            current_wall_norms = []
            current_wall_texcs = []

            def gen_seg_poly(edge_p0, side_vec, seg_start, seg_end, min_y, max_y):
                if seg_end == seg_start:
                    return
                if min_y == max_y:
                    return

                s_p0 = edge_p0 + seg_start * side_vec
                s_p1 = edge_p0 + seg_end * side_vec

                # If this polygon starts at ground level, add a collidable segment
                if min_y == 0:
                    self.wall_segs.append(np.array([s_p1, s_p0]))

                # Generate the vertices
                # Vertices are listed in counter-clockwise order
                current_wall_verts.append(s_p0 + min_y * Y_VEC)
                current_wall_verts.append(s_p0 + max_y * Y_VEC)
                current_wall_verts.append(s_p1 + max_y * Y_VEC)
                current_wall_verts.append(s_p1 + min_y * Y_VEC)

                # Compute the normal for the polygon
                normal = np.cross(s_p1 - s_p0, Y_VEC)
                normal = -normal / np.linalg.norm(normal)
                for _ in range(4):
                    current_wall_norms.append(normal)

                # Generate the texture coordinates
                texcs = gen_texcs_wall(
                    tex, seg_start, min_y, seg_end - seg_start, max_y - min_y
                )
                current_wall_texcs.append(texcs)

            # Logic to call gen_seg_poly (adapted from Room)
            edge_p0 = self.outline[wall_idx, :]
            edge_p1 = self.outline[(wall_idx + 1) % self.num_walls, :]
            wall_width = np.linalg.norm(edge_p1 - edge_p0)
            side_vec = (edge_p1 - edge_p0) / wall_width

            if len(self.portals[wall_idx]) > 0:
                seg_end = self.portals[wall_idx][0]["start_pos"]
            else:
                seg_end = wall_width

            # Generate the first polygon (going up to the first portal)
            gen_seg_poly(edge_p0, side_vec, 0, seg_end, 0, self.wall_height)

            # For each portal in this wall
            for portal_idx, portal in enumerate(self.portals[wall_idx]):
                portal = self.portals[wall_idx][portal_idx]
                start_pos = portal["start_pos"]
                end_pos = portal["end_pos"]
                min_y = portal["min_y"]
                max_y = portal["max_y"]

                # Generate the bottom polygon
                gen_seg_poly(edge_p0, side_vec, start_pos, end_pos, 0, min_y)

                # Generate the top polygon
                gen_seg_poly(
                    edge_p0, side_vec, start_pos, end_pos, max_y, self.wall_height
                )

                if portal_idx < len(self.portals[wall_idx]) - 1:
                    next_portal = self.portals[wall_idx][portal_idx + 1]
                    next_portal_start = next_portal["start_pos"]
                else:
                    next_portal_start = wall_width

                # Generate the polygon going up to the next portal
                gen_seg_poly(
                    edge_p0, side_vec, end_pos, next_portal_start, 0, self.wall_height
                )

            # Store data for this wall
            if len(current_wall_verts) > 0:
                self.wall_render_data.append(
                    {
                        "tex": tex,
                        "verts": np.array(current_wall_verts),
                        "norms": np.array(current_wall_norms),
                        "texcs": np.concatenate(current_wall_texcs)
                        if len(current_wall_texcs) > 0
                        else np.array([]),
                    }
                )

        if len(self.wall_segs) > 0:
            self.wall_segs = np.array(self.wall_segs)
        else:
            self.wall_segs = np.array([]).reshape(0, 2, 3)

    def _render(self):
        """
        Render the static elements of the room
        """
        glColor3f(1, 1, 1)

        # Draw the floor
        self.floor_tex.bind()
        glBegin(GL_POLYGON)
        glNormal3f(0, 1, 0)
        for i in range(self.floor_verts.shape[0]):
            glTexCoord2f(*self.floor_texcs[i, :])
            glVertex3f(*self.floor_verts[i, :])
        glEnd()

        # Draw the ceiling
        if not self.no_ceiling:
            self.ceil_tex.bind()
            glBegin(GL_POLYGON)
            glNormal3f(0, -1, 0)
            for i in range(self.ceil_verts.shape[0]):
                glTexCoord2f(*self.ceil_texcs[i, :])
                glVertex3f(*self.ceil_verts[i, :])
            glEnd()

        # Draw the walls
        for wall_data in self.wall_render_data:
            wall_data["tex"].bind()
            glBegin(GL_QUADS)
            verts = wall_data["verts"]
            norms = wall_data["norms"]
            texcs = wall_data["texcs"]
            for i in range(verts.shape[0]):
                glNormal3f(*norms[i, :])
                glTexCoord2f(*texcs[i, :])
                glVertex3f(*verts[i, :])
            glEnd()


class InvisibleBox(Box):
    def render(self):
        pass


class WaterMaze(MiniWorldEnv, utils.EzPickle):
    """
    ## Description
    Environment with 4 differently colored/textured walls.
    Sparse reward: +1 when reaching the red box, 0 otherwise.
    Predefined start positions.
    """

    def __init__(self, size=10, max_episode_steps=180, **kwargs):
        assert size >= 2
        self.size = size

        # Predefined start positions (x, z)
        # Assuming room is from (0,0) to (size, size)
        self.start_positions = [
            (2.0, 2.0),
            (size - 2.0, 2.0),
            (2.0, size - 2.0),
            (size - 2.0, size - 2.0),
        ]

        MiniWorldEnv.__init__(self, max_episode_steps=max_episode_steps, **kwargs)
        utils.EzPickle.__init__(
            self, size=size, max_episode_steps=max_episode_steps, **kwargs
        )

        # Allow only movement actions (left/right/forward)
        self.action_space = spaces.Discrete(self.actions.move_forward + 1)

    def _gen_world(self):
        # Define 4 textures for the 4 walls
        # Order: East, North, West, South (based on add_rect_room implementation)
        wall_textures = ["brick_wall", "wood", "concrete", "drywall"]

        # Create the room manually to use FourColoredRoom
        # add_rect_room logic:
        min_x, max_x = 0, self.size
        min_z, max_z = 0, self.size

        outline = np.array(
            [
                [max_x, max_z],  # East
                [max_x, min_z],  # North
                [min_x, min_z],  # West
                [min_x, max_z],  # South
            ]
        )

        room = FourColoredRoom(wall_textures=wall_textures, outline=outline)
        self.rooms.append(room)

        # Place the goal (red box) at a fixed position (center of the room)
        self.box = self.place_entity(
            InvisibleBox(color="red"), pos=np.array([self.size - 4, 0, 4])
        )

        # Place agent at one of the predefined positions
        start_pos_idx = self.np_random.integers(0, len(self.start_positions))
        start_pos = self.start_positions[start_pos_idx]

        # Add some random orientation
        start_dir = self.np_random.uniform(0, 2 * math.pi)

        self.place_agent(dir=start_dir, pos=np.array([start_pos[0], 0, start_pos[1]]))

    def step(self, action):
        obs, reward, termination, truncation, info = super().step(action)

        # Sparse reward: 1 if reached goal, 0 otherwise
        if self.near(self.box):
            reward = 1.0
            termination = True
        else:
            reward = 0.0

        return obs, reward, termination, truncation, info


# # Register the environment
# gym.register(
#     id="MiniWorld-WaterMaze-v0",
#     entry_point="miniworld.envs.water_maze:WaterMaze",
# )
