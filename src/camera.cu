#include <camera.cuh>

void Camera::initialize()
{
    _img_height = int(img_width / aspect_ratio);
    _img_height = (_img_height < 1) ? 1 : _img_height;

    _camera_center = Point(0, 0, 0);

    auto focal_length = 1.0;
    auto viewport_height = 2.0;
    auto viewport_width = viewport_height * (double(img_width)/_img_height);

    auto viewport_u = Vector(viewport_width, 0, 0);
    auto viewport_v = Vector(0, -viewport_height, 0);

    _pixel_delta_u = viewport_u / img_width;
    _pixel_delta_v = viewport_v / _img_height;

    auto viewport_upper_left = _camera_center
                             - Vector(0, 0, focal_length) - viewport_u/2 - viewport_v/2;
    _pixel00_loc = viewport_upper_left + 0.5 * (_pixel_delta_u + _pixel_delta_v);

    _scene_info = SceneInfo{_pixel00_loc, _camera_center, _pixel_delta_u, _pixel_delta_v};

    // printf("Camera::initialize: scene_info: pixel00_loc (%f, %f, %f)", _scene_info.pixel00_loc.x(),_scene_info.pixel00_loc.y(), _scene_info.pixel00_loc.z());
}

int Camera::get_img_height() const
{
    return _img_height;
}

SceneInfo Camera::get_scene_info() const
{
    return _scene_info;
}