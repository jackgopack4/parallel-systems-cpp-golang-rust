#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <GL/glu.h>
#include <GL/glut.h>
#include <iostream>
#include <vector>

const int MAX_CAPACITY = 1;
struct Point {
    int index;
    double x, y;
    double x_vel, y_vel;
    double mass;
    Point(int _idx, double _x, double _y) : index(_idx), x(_x), y(_y), x_vel(0.0), y_vel(0.0), mass(-1) {}
    Point(int _idx, double _x, double _y, double _mass, double _x_vel, double _y_vel) : index(_idx), x(_x), y(_y), x_vel(_x_vel), y_vel(_y_vel), mass(_mass) {
        if (_x >= 4 || _x <= 0 || y >= 4 || y <= 0) {
            mass = -1.;
        }
    }
};
void printPoint(const Point& point);

struct Node {
    double centerX, centerY;
    double width, height;
    std::vector<Point> points;
    Node* children[4];  // NW, NE, SW, SE

    Node(double x, double y, double w, double h) : centerX(x), centerY(y), width(w), height(h) {
        for (int i = 0; i < 4; ++i) {
            children[i] = nullptr;
        }
    }

    ~Node() {
        for (int i = 0; i < 4; ++i) {
            delete children[i];
        }
    }

    bool isLeaf() const {
        return children[0] == nullptr;
    }

};

class Quadtree {
public:
    Quadtree(double w, double h) : root(new Node(w / 2, h / 2, w, h)) {}

    ~Quadtree() {
        delete root;
    }

    void insert(const Point& point) {
        insert(root, point);
    }

    void insert(Node* node, const Point& point) {
      std::cout << "inserting point (" << point.x << ", " << point.y << ")" << std::endl;
      if (point.x > 0 && point.x < 4 && point.y > 0 && point.y < 4) {
        if (node->isLeaf()) {
            if (node->points.size() < MAX_CAPACITY) {
                node->points.push_back(point);
            } else {
                split(node);
                insert(node, point);
            }
        } else {
            int index = getIndex(node, point);
            insert(node->children[index], point);
        }
      }

    }

    void printTree() const {
        printTree(root, 0);
    }

    void printTree(int depth)const {
        printTree(root,depth);
    }

private:
    void split(Node* node) {
        double newWidth = node->width / 2;
        double newHeight = node->height / 2;

        node->children[0] = new Node(node->centerX - newWidth / 2, node->centerY - newHeight / 2, newWidth, newHeight);
        node->children[1] = new Node(node->centerX + newWidth / 2, node->centerY - newHeight / 2, newWidth, newHeight);
        node->children[2] = new Node(node->centerX - newWidth / 2, node->centerY + newHeight / 2, newWidth, newHeight);
        node->children[3] = new Node(node->centerX + newWidth / 2, node->centerY + newHeight / 2, newWidth, newHeight);

        for (const Point& p : node->points) {
            int index = getIndex(node, p);
            insert(node->children[index], p);
        }

        node->points.clear();
    }

    int getIndex(Node* node, const Point& point) const {
        int index = 0;
        if (point.x > node->centerX) index += 1;
        if (point.y > node->centerY) index += 2;
        return index;
    }

    void printTree(const Node* node, int depth) const {
        if (node) {
            // Print information about the current node
            for (int i = 0; i < depth; ++i) {
                std::cout << "  ";
            }
            std::cout << "Node (" << node->centerX << ", " << node->centerY << ") width: " << node->width << ", height: " << node->height << " ";
            if (node->points.size()) {
              printPoint(node->points[0]);
            }
            std::cout << std::endl;
            // Recursively print the children
            for (int i = 0; i < 4; ++i) {
                printTree(node->children[i], depth + 1);
            }
        }
    }

private:
    Node* root;
};

void printPoint(const Point& point) {
    std::cout << "Point - Index: " << point.index << ", "
              << "Position: (" << point.x << ", " << point.y << "), "
              << "Velocity: (" << point.x_vel << ", " << point.y_vel << "), "
              << "Mass: " << point.mass;
}

int main() {
    Quadtree quadtree(4, 4);
    quadtree.printTree();
    // Insert some points for testing
    quadtree.insert(Point(5,	2.810173e+00,	1.999232e+00,	2.235070e+00,	0.000000e+00,	0.000000e+00));
    quadtree.insert(Point(0,    2.054482e+00,	6.003183e-01,	2.647132e+00,	0.000000e+00,	0.000000e+00));
    //quadtree.printTree();
    quadtree.insert(Point(1,	2.647132e+00,	3.554528e+00,	2.831508e+00,	0.000000e+00,	0.000000e+00));
    //quadtree.printTree();
    quadtree.insert(Point(7,	3.985606e+00,	1.332045e-01,	8.786317e-01,	0.000000e+00,	0.000000e+00));
    quadtree.insert(Point(2,	2.831508e+00,	2.876292e+00,	2.996487e+00,	0.000000e+00,	0.000000e+00));
    quadtree.insert(Point(3,	2.996487e+00,	1.272099e+00,	2.288692e+00,	0.000000e+00,	0.000000e+00));
    quadtree.insert(Point(4,	2.288692e+00,	2.443186e+00,	2.810173e+00,	0.000000e+00,	0.000000e+00));
    quadtree.insert(Point(9,	8.985912e-01,	1.985868e+00,	1.896852e+00,	0.000000e+00,	0.000000e+00));
    quadtree.insert(Point(6,	2.235070e+00,	2.731189e+00,	3.985606e+00,	0.000000e+00,	0.000000e+00));
    quadtree.insert(Point(8,	8.786317e-01,	3.940775e+00,	8.985912e-01,	0.000000e+00,	0.000000e+00));
    quadtree.printTree();
    return 0;
}
