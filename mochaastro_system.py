"""Star and planetary system objects for mochaastro.py"""
from math import hypot
from mochaastro_body import Body
from mochaastro_common import axisEqual3D

class System:
	"""Set of orbiting bodies"""
	def __init__(self, parent: Body, *bodies: Body) -> None:
		"""Star system containing bodies.\nDoesn't need to be ordered."""
		self.parent = parent
		self.bodies = set(bodies)

	@property
	def sorted_bodies(self) -> list:
		"""List of bodies sorted by semimajor axis"""
		return sorted(list(self.bodies), key=lambda x: x.orbit.a)

	def plot(self) -> None:
		"""Plot system with pyplot"""
		import matplotlib.pyplot as plt
		# see above plot for notes and sources
		n = 1000

		plt.figure(figsize=(7, 7))
		ax = plt.axes(projection='3d')
		ax.set_title('Orbit')
		ax.set_xlabel('x (m)')
		ax.set_ylabel('y (m)')
		ax.set_zlabel('z (m)')
		ax.scatter(0, 0, 0, marker='*', color='y', s=50, zorder=2)
		ax.text(0, 0, 0, self.parent.name, size=8, color='k', zorder=4)
		for body in self.bodies:
			cs = [body.orbit.cartesian(t*body.orbit.p/n) for t in range(n)]
			xs, ys, zs, _, __, ___ = zip(*cs)
			ax.plot(xs, ys, zs, color='k', zorder=1)
			ax.scatter(xs[0], ys[0], zs[0], marker='o', s=15, zorder=3)
			ax.text(xs[0], ys[0], zs[0], body.name, size=8, color='k', zorder=4)

		axisEqual3D(ax)
		plt.show()

	def plot2d(self) -> None:
		"""2D Plot system with pyplot"""
		import matplotlib.pyplot as plt
		# see above plot for notes and sources
		n = 1000

		plt.figure(figsize=(7, 7))
		plt.title('Orbit')
		plt.xlabel('x (m)')
		plt.ylabel('y (m)')
		plt.scatter(0, 0, marker='*', color='y', s=50, zorder=2)
		for body in self.bodies:
			cs = [body.orbit.cartesian(t*body.orbit.p/n) for t in range(n)]
			xs, ys, _, __, ___, ____ = zip(*cs)
			plt.plot(xs, ys, color='k', zorder=1)
			plt.scatter(xs[0], ys[0], marker='o', s=15, zorder=3)

		plt.show()

	def mass_pie(self) -> None:
		"""Mass pie chart"""
		import matplotlib.pyplot as plt
		system_masses = [i.mass for i in self.bodies]

		plt.subplot(1, 2, 1)
		plt.title('System mass')
		plt.pie(system_masses + [self.parent.mass])

		plt.subplot(1, 2, 2)
		plt.title('System mass (excl. primary)')
		plt.pie(system_masses)

		plt.show()

	def grav_sim(self, focus: Body = None, res = 10000, p=25, skip=100) -> None:
		"""Model newtonian gravity"""
		import matplotlib.pyplot as plt
		# from time import time
		# solar_system_object.grav_sim()
		parent = self.parent
		bodies = list(self.bodies) + [parent]
		steps = res*p
		timestep = (focus.orbit.p if focus else min(b.orbit.p for b in self.bodies)) / res

		body_xv = [body.orbit.cartesian() if body != parent else (0, 0, 0, 0, 0, 0) for body in bodies]
		body_x = [list(elem[:3]) for elem in body_xv]
		body_v = [list(elem[3:]) for elem in body_xv]
		body_a = [[0, 0, 0] for _ in bodies]
		body_mu_cache = [b.mu for b in bodies]
		xs = []
		# start = time()
		# compute positions
		for i in range(steps):
			xs.append([])
			for (bi, body) in enumerate(bodies):
				if (focus and body != focus):
					xs[i].append((0, 0, 0) if body == parent else body.orbit.cartesian(timestep * i)[:3])
					continue
				# copy xs to output list
				xs[i].append(body_x[bi][:])
				# new x, v
				for dim in range(3):
					body_x[bi][dim] += body_v[bi][dim] * timestep
					body_v[bi][dim] += body_a[bi][dim] * timestep
				# reset acc...
				body_a[bi] = [0, 0, 0]
				# new acc
				for (other_body_i, other_body_x) in enumerate(body_x):
					if other_body_i == bi:
						continue
					dx = tuple(other_body_x[dim] - body_x[bi][dim] for dim in range(3))
					r = hypot(*dx)
					a = body_mu_cache[other_body_i] / (r*r)
					for dim in range(3):
						body_a[bi][dim] += a * dx[dim] / r
		# print(time() - start)
		# draw
		plt.figure(figsize=(7, 7))
		ax = plt.axes(projection='3d')
		ax.set_title('Orbit')
		ax.set_xlabel('x (m)')
		ax.set_ylabel('y (m)')
		ax.set_zlabel('z (m)')
		# ax.scatter(0, 0, 0, marker='*', color='y', s=50, zorder=2)
		for (p_i, planet_coords) in enumerate(list(zip(*xs))):
			x, y, z = zip(*planet_coords[::skip])
			ax.plot(x, y, z, color='k', zorder=p_i)
			ax.scatter(x[0], y[0], z[0], marker='o', s=15, zorder=len(bodies) + p_i)
			ax.scatter(x[-1], y[-1], z[-1], marker='o', s=15, zorder=2*len(bodies) + p_i)
			ax.text(*planet_coords[0], bodies[p_i].name + "^", \
				size=8, color='k', zorder=3*len(bodies) + p_i)
			ax.text(*planet_coords[-1], bodies[p_i].name + "$", \
				size=8, color='k', zorder=4*len(bodies) + p_i)

		axisEqual3D(ax)
		plt.show()
