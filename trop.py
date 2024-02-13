import itertools as it
from typing import Dict, Tuple

import numpy as np
from scipy.spatial import ConvexHull

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
plt.style.use('fivethirtyeight')


class TropicalPolynomial:
    def __init__(self, polynomial: Dict[Tuple[int], float]):
        self.polynomial = polynomial
        self.tropical_expression = ' + '.join(str(coeff) + ' x^' + str(deg[0]) + ' y^' + str(deg[1]) for deg, coeff in self.polynomial.items())
        self.newton_polygon = self.compute_newton_polygon()
        self.newton_polytope = self.compute_newton_polytope()
        self.subdivision = self.compute_subdivision()

    def __repr__(self):
        return self.polynomial

    def tropical_representation(self):
        """
        Returns the traditional representation for the polynomial where the 
        operations are understood to be tropical.
        """
        expression = ''
        num_terms = len(self.polynomial.items())
        for i, (deg, coeff) in enumerate(self.polynomial.items()):
            # coefficient cases
            subexpression = ''
            if coeff != 0:
                subexpression += f'{coeff}'
            else:
                # degree 0 term
                if not any(deg):
                    subexpression += f'{coeff}'
            # degree of x cases
            if deg[0] != 0:
                # if coeff != 0:
                #     subexpression += '+'
                if deg[0] == 1:
                    subexpression += 'x'
                else:
                    subexpression += f'x^{deg[0]}'
            # degree of y cases
            if deg[1] != 0:
                # if coeff != 0:
                #     subexpression += '+'
                if deg[1] == 1:
                    subexpression += 'y'
                else:
                    subexpression += f'y^{deg[1]}'
            # join terms together
            if i != num_terms-1:
                expression += subexpression + ' + '
            else:
                expression += subexpression
        return f'{expression}'

    def classical_representation(self):
        """
        Returns the representation for the polynomial with classical operations.
        """
        expression = ''
        num_terms = len(self.polynomial.items())
        for i, (deg, coeff) in enumerate(self.polynomial.items()):
            # coefficient cases
            subexpression = ''
            if coeff != 0:
                subexpression += f'{coeff}'
            else:
                # degree 0 term
                if not any(deg):
                    subexpression += f'{coeff}'
            # degree of x cases
            if deg[0] != 0:
                if coeff != 0:
                    subexpression += '+'
                if deg[0] == 1:
                    subexpression += 'x'
                else:
                    subexpression += f'{deg[0]}x'
            # degree of y cases
            if deg[1] != 0:
                if coeff != 0:
                    subexpression += '+'
                if deg[1] == 1:
                    subexpression += 'y'
                else:
                    subexpression += f'{deg[1]}y'
            # join terms together
            if i != num_terms-1:
                expression += subexpression + ', '
            else:
                expression += subexpression
        return f'max({expression})'

    def compute_newton_polygon(self):
        points = np.vstack([deg for deg in self.polynomial.keys()])
        return ConvexHull(points, incremental=False)

    def compute_newton_polytope(self):
        # compute higher dimensional Newton polytope
        lifted_points = np.vstack([np.hstack([deg, coeff]) for deg, coeff in self.polynomial.items()])
        return ConvexHull(lifted_points, incremental=False)

    def compute_subdivision(self):
        """Computes the dual subdivision of the Newton polygon of the tropical polynomial.
        
        Refer to https://arxiv.org/pdf/math/0306366.pdf (page 11) for an algorithm
        to compute the dual subdivision of the tropical polynomial.
        """        
        # print(newton_polytope.points[newton_polytope.vertices])
        barycenter = np.mean(self.newton_polytope.points[self.newton_polytope.vertices], axis=0)
        triangles = []
        for simplex in self.newton_polytope.simplices:
            triangle = self.newton_polytope.points[simplex]
            normal = np.cross(triangle[1, :] - triangle[0, :], triangle[2, :] - triangle[0, :])
            # reorient normal
            normal = normal if normal @ (triangle[0, :] - barycenter) > 0 else -normal
            if normal[-1] > 0:
                triangles.append(triangle[:, :2].astype(int))
        return triangles

    def plot_upper_envelope(self, with_normals: bool=False):
        """Plots the upper envelope of the tropical polynomial.

        The upper envelope is defined as the upper faces of the convex hull of
        the Newton polytope.
        
        Parameters
        ----------
        with_normals : bool, optional
            Plot the upper envelope with the normal vectors of the faces
            pointing out.
        """
        # set up plot
        ax = plt.gca(projection='3d')
        ax.set_xlim([np.min(self.newton_polytope.points[:, 0])-1, np.max(self.newton_polytope.points[:, 0])+1])
        ax.set_ylim([np.min(self.newton_polytope.points[:, 1])-1, np.max(self.newton_polytope.points[:, 1])+1])
        ax.set_zlim([np.min(self.newton_polytope.points[:, 2])-1, np.max(self.newton_polytope.points[:, 2])+1])

        # plot triangles
        ax.add_collection(Poly3DCollection([self.newton_polytope.points[simplex] for simplex in self.newton_polytope.simplices], alpha=0.5))
        
        # plot edges
        for simplex in self.newton_polytope.simplices:
            triangle = self.newton_polytope.points[simplex]
            for vertex_subset in [[0,1], [0,2], [1,2]]:
                ax.plot(triangle[vertex_subset, 0], triangle[vertex_subset, 1], triangle[vertex_subset, 2], color='r')
    
        # plot vertices
        ax.scatter(self.newton_polytope.points[:, 0], self.newton_polytope.points[:, 1], self.newton_polytope.points[:, 2], color='k')

        # plot normal vectors
        if with_normals:
            barycenter = np.mean(self.newton_polytope.points[self.newton_polytope.vertices], axis=0)
            for simplex in self.newton_polytope.simplices:
                triangle = self.newton_polytope.points[simplex]
                normal = np.cross(triangle[1, :] - triangle[0, :], triangle[2, :] - triangle[0, :])
                # reorient normal (https://math.stackexchange.com/a/434420)
                normal = normal if normal @ (triangle[0, :] - barycenter) > 0 else -normal

                # move normal vector to face
                triangle_barycenter = np.mean(triangle, axis=0)
                ax.quiver(triangle_barycenter[0], triangle_barycenter[1], triangle_barycenter[2], normal[0], normal[1], normal[2], length=1, normalize=True, color='gray', alpha=0.5)
        plt.show()

    def plot_subdivision(self, with_labels: bool=True):
        """Plots the dual Newton subdivision of the tropical polynomial.
        
        Parameters
        ----------
        with_labels : bool, optional
            Plot vertices with labels.
        """
        # set up plot
        fig, ax = plt.subplots()
        ax.set_title(f'$f(x,y) = {self.tropical_representation()}$')
        
        # set plot bounds
        degrees = [sum(deg) for deg in self.polynomial.keys()]
        xmin, xmax = min(degrees) - 1, max(degrees) + 1
        ymin, ymax = xmin, xmax
        ax.set_xlim((xmin, xmax))
        ax.set_ylim((ymin, ymax))
        ax.set_xticks(np.arange(xmin, xmax))
        ax.set_yticks(np.arange(ymin, ymax))

        # plot triangles in subdivision
        for i, triangle in enumerate(self.subdivision, start=1):
            for vertex_subset in [[0,1], [0,2], [1,2]]:
                ax.plot(triangle[vertex_subset, 0], triangle[vertex_subset, 1], color='b', zorder=1)
                if with_labels:
                    barycenter = triangle.mean(axis=0)
                    ax.text(barycenter[0], barycenter[1], f'{i}')
        
        # plot vertices in subdivision
        lattice_points = np.array([(a,b) for (a,b) in it.product(range(xmin+1, xmax), range(ymin+1, ymax))])
        ax.scatter(lattice_points[:, 0], lattice_points[:, 1], color='gray', zorder=2)
        poly_points = np.array([(a,b) for (a,b) in self.polynomial.keys()])
        ax.scatter(poly_points[:, 0], poly_points[:, 1], color='k', zorder=3)
        plt.show()

    def plot_curve(self, margin: float=2, with_labels: bool=True):
        """Plots the tropical curve in the plane.
        
        Parameters
        ----------
        margin : float, optional
            How much to extend the plot beyond the outermost vertices.
            The default value is 2.
        with_labels : bool, optional
            Plot vertices with labels.
        """
        # set up plot
        fig, ax = plt.subplots()
        ax.set_title(f'$f(x,y) = {self.tropical_representation()}$')

        # find vertices
        # NOTE: index into vertices is the same as the index into triangles of self.subdivision
        # vertices is a dict with vertex/exponents key/values
        vertices = dict()
        for i, triangle in enumerate(self.subdivision, start=1):
            # set up system of equations
            exponents = [tuple(vertex) for vertex in triangle]
            A_1, b_1 = (triangle[0, :] - triangle[1, :]).reshape(1,2), np.array(self.polynomial[exponents[1]] - self.polynomial[exponents[0]]).reshape((1,1))
            A_2, b_2 = (triangle[1, :] - triangle[2, :]).reshape(1,2), np.array(self.polynomial[exponents[2]] - self.polynomial[exponents[1]]).reshape((1,1))
            A, b = np.vstack([A_1, A_2]), np.vstack([b_1, b_2])
            vertex = np.linalg.solve(A, b)
            if with_labels:
                ax.text(vertex[0], vertex[1], f'{i}')

            hashed_vertex = tuple(np.squeeze(vertex))
            vertices[hashed_vertex] = set(exponents)
        all_vertices = np.vstack(list(vertices.keys()))
        ax.scatter(all_vertices[:, 0], all_vertices[:, 1], color='k', zorder=2)
        
        # set up plot limits
        x_min, x_max = np.min(all_vertices[:, 0]) - margin, np.max(all_vertices[:, 0]) + margin
        y_min, y_max = np.min(all_vertices[:, 1]) - margin, np.max(all_vertices[:, 1]) + margin
        ax.set_xlim((x_min, x_max))
        ax.set_ylim((y_min, y_max))

        # plot edges
        all_dual_edges = {(exponent1, exponent2) for exponents in vertices.values() for (exponent1, exponent2) in it.combinations(exponents, 2)}
        edge_is_bounded = {edge: False for edge in all_dual_edges}
        # find bounded edges
        for (vertex1, vertex2) in it.combinations(vertices, 2):
            common_exponents = list(vertices[tuple(vertex1)].intersection(vertices[tuple(vertex2)]))
            if len(common_exponents) == 2:
                # edge is not directed
                edge_is_bounded[(common_exponents[0], common_exponents[1])] = True
                edge_is_bounded[(common_exponents[1], common_exponents[0])] = True
                points = np.vstack([np.array(vertex1), np.array(vertex2)])
                ax.plot(points[:, 0], points[:, 1], color='b', zorder=1)
        # find unbounded edges
        for i, vertex in enumerate(vertices, start=1):            
            triangle = np.vstack(list(vertices[vertex]))
            barycenter = np.mean(triangle, axis=0)
            edges = [(tuple(triangle[subset[0], :]), tuple(triangle[subset[1], :])) for subset in [[0,1], [0,2], [1,2]]]
            for edge in edges:
                # unbounded edges
                is_bounded = edge_is_bounded[edge]
                if not is_bounded:
                    # avoid division by zero for slope
                    if edge[0][1] != edge[1][1]:
                        slope = -(edge[0][0] - edge[1][0]) / (edge[0][1] - edge[1][1])
                        midpoint = np.mean(np.array(np.vstack(edge)), axis=0)
                        if np.array([1, slope]) @ (midpoint - barycenter) > 0:
                            points = np.vstack([vertex, np.array([x_max, slope*(x_max - vertex[0]) + vertex[1]])])
                        else:
                            points = np.vstack([vertex, np.array([x_min, slope*(x_min - vertex[0]) + vertex[1]])])
                    # vertical line
                    else:
                        midpoint = np.mean(np.array(np.vstack(edge)), axis=0)
                        if np.array([0, 1]) @ (midpoint - barycenter) > 0:
                            points = np.vstack([vertex, np.array([vertex[0], y_max])])
                        else:
                            points = np.vstack([vertex, np.array([vertex[0], y_min])])
                    ax.plot(points[:, 0], points[:, 1], color='b', zorder=1)
        plt.show()


def test_code(polynomial):
    print(f'p(x) = {polynomial.tropical_representation()}')
    print(f'p(x) = {polynomial.classical_representation()}')
    print()
    polynomial.plot_upper_envelope(with_normals=False)
    polynomial.plot_subdivision()
    polynomial.plot_curve()
    print('\n'*3)


def main():
    f_3a = {(0,0): 0.5,
            (1,0): 2,
            (0,1): -5}
    # trop_f_3a = TropicalPolynomial(f_3a)
    # test_code(trop_f_3a)

    f_3b = {(0,0): 3,
            (1,0): 2,
            (0,1): 2,
            (1,1): 3,
            (0,2): 0,
            (2,0): 0}
    trop_f_3b = TropicalPolynomial(f_3b)
    test_code(trop_f_3b)

    f_3c = {(0,0): 0,
            (1,0): 0,
            (0,1): 0,
            (0,2): 0,
            (2,0): -1}
    trop_f_3c = TropicalPolynomial(f_3c)
    test_code(trop_f_3c)

    p = {(0,0): 5,
         (1,0): 5,
         (0,1): 5,
         (1,1): 4,
         (0,2): 1,
         (2,0): 0}
    trop_p = TropicalPolynomial(p)
    test_code(trop_p)
    
    q = {(0,0): 7,
         (1,0): 4,
         (0,1): 0,
         (1,1): 4,
         (0,2): 3,
         (2,0): -3}
    trop_q = TropicalPolynomial(q)
    test_code(trop_q)

    f_1 = {(0,0): 0,
           (1,0): 0,
           (0,1): 0,
           (1,1): -1}
    trop_f_1 = TropicalPolynomial(f_1)
    test_code(trop_f_1)

    f_2 = {(0,0): 0,
           (-1,0): 0,
           (0,-1): 0,
           (1,0): -1,
           (0,1): -2}
    trop_f_2 = TropicalPolynomial(f_2)
    test_code(trop_f_2)

    f_3 = {(0,0): 0,
           (1,0): 0,
           (0,1): 0,
           (2,0): -3,
           (0,2): -4,
           (1,1): -1}
    trop_f_3 = TropicalPolynomial(f_3)
    # test_code(trop_f_3)

    f_4 = {(0,0): 0,
           (1,0): 0,
           (1,1): -1,
           (1,2): 0,
           (2,1): -2}
    trop_f_4 = TropicalPolynomial(f_4)
    test_code(trop_f_4)


if __name__ == '__main__':
    main()