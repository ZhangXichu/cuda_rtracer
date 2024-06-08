#include <camera.cuh>

__device__ Hittable** sphere_lst;
__device__ Hittable* world;

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

__device__ Color Camera::ray_color(curandState* rand_states, int max_depth, const Ray& ray)
{
    Color accumulated_color(1.0, 1.0, 1.0); 
    Ray current_ray = ray;
    int depth = 0;

    HitRecord record;

    Sphere sphere(Point(0, 0, -1), 0.5);
    Sphere sphere2(Point(0,-100.5,-1), 100);

    while (depth < max_depth) {
        
        if (world->hit(current_ray, Interval(0.0, infinity), record))
        {
            Vector direction = random_on_hemisphere(rand_states, record.normal);
            current_ray = Ray(record.p, direction);
            accumulated_color *= 0.5;
            
        } else {
            
            Vector unit_direction = unit_vector(current_ray.direction());
            auto a = 0.5*(unit_direction.y() + 1.0);
            Color background = (1.0 - a) * Color(1.0, 1.0, 1.0) + a * Color(0.5, 0.7, 1.0);

            accumulated_color = accumulated_color * background;
            break;
        }
        
        depth++;
    }
    return accumulated_color;
}

int Camera::get_img_height() const
{
    return _img_height;
}

SceneInfo Camera::get_scene_info() const
{
    return _scene_info;
}