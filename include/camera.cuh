#ifndef CMAERA_H
#define CAMERA_H

#include <vector.cuh>

struct SceneInfo {
    Vector pixel00_loc;
    Vector camera_center;
    Vector pixel_delta_u;
    Vector pixel_delta_v;
};


class Camera {

public:
    double aspect_ratio = 16.0 / 9.0;
    int img_width = 800;

    int get_img_height() const;
    SceneInfo get_scene_info() const;

    void initialize();


private:
    int _img_height;
    Point _camera_center;
    Point _pixel00_loc;
    Vector _pixel_delta_u; // Offset to pixel to the right
    Vector _pixel_delta_v; // Offset to pixel below
    SceneInfo _scene_info;
};


#endif
