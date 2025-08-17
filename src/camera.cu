#include <camera.cuh>

__device__ Hittable** obj_lst;
__device__ Hittable* world;

void Camera::initialize()
{
    _img_height = int(img_width / aspect_ratio);
    _img_height = (_img_height < 1) ? 1 : _img_height;

    // _camera_center = Point(0, 0, 0);
    _camera_center = lookfrom;

    // auto focal_length = 1.0;
    auto focal_length = (lookfrom - lookat).length();
    auto theta = degrees_to_radians(vfov);
    auto h = tan(theta/2);
    // auto viewport_height = 2 * h * focal_length;
    auto viewport_height = 2 * h * focus_dist;
    auto viewport_width = viewport_height * (double(img_width)/_img_height);

    // Calculate the u,v,w unit basis vectors for the camera coordinate frame.
    _w = unit_vector(lookfrom - lookat);
    _u = unit_vector(cross(vup, _w));
    _v = cross(_w, _u);

    // Calculate the vectors across the horizontal and down the vertical viewport edges.
    Vector viewport_u = viewport_width * _u;    // Vector across viewport horizontal edge
    Vector viewport_v = viewport_height * -_v;  // Vector down viewport vertical edge

    _pixel_delta_u = viewport_u / img_width;
    _pixel_delta_v = viewport_v / _img_height;

    auto viewport_upper_left = _camera_center - (focus_dist * _w) - viewport_u/2 - viewport_v/2;
    _pixel00_loc = viewport_upper_left + 0.5 * (_pixel_delta_u + _pixel_delta_v);

    // Calculate the camera defocus disk basis vectors.
    auto defocus_radius = focus_dist * tan(degrees_to_radians(defocus_angle / 2));
    _defocus_disk_u = _u * defocus_radius;
    _defocus_disk_v = _v * defocus_radius;

    _scene_info = SceneInfo{_pixel00_loc, _camera_center, _pixel_delta_u, _pixel_delta_v};
}

__device__ Color Camera::ray_color(curandState* rand_states, int max_depth, const Ray& ray)
{
    Color accumulated_color(1.0, 1.0, 1.0); 
    Ray current_ray = ray;
    int depth = 0;

    HitRecord record;

    while (depth < max_depth) {
        
        if (world->hit(current_ray, Interval(0.001, infinity), record))
        {
            // Vector direction = record.normal + random_unit_vector(rand_states);
            Ray scattered;
            Color attenuation;
            if (record.material->scatter(current_ray, record, attenuation, scattered, rand_states))
            {
                accumulated_color = accumulated_color * attenuation;
                current_ray = scattered;
            }
            
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

__device__ Vector Camera::sample_square(curandState* rand_states) {
        return Vector(random_double(rand_states) - 0.5, random_double(rand_states) - 0.5, 0);
    }

__device__ Ray Camera::get_ray(int i, int j, curandState* rand_states)
{
    // Construct a camera ray originating from the defocus disk and directed at a randomly
        // sampled point around the pixel location i, j.
    auto offset = sample_square(rand_states);
    auto pixel_sample = _scene_info.pixel00_loc
                        + ((i + offset.x()) * _scene_info.pixel_delta_u)
                        + ((j + offset.y()) * _scene_info.pixel_delta_v);

    // auto ray_origin = _scene_info.camera_center;
    auto ray_origin = (defocus_angle <= 0) ? _scene_info.camera_center : defocus_disk_sample(rand_states);
    auto ray_direction = pixel_sample - ray_origin;
    auto ray_time = random_double(rand_states);

    return Ray(ray_origin, ray_direction, ray_time);
}

int Camera::get_img_height() const
{
    return _img_height;
}

SceneInfo Camera::get_scene_info() const
{
    return _scene_info;
}

__device__ Point Camera::defocus_disk_sample(curandState* rand_states) const 
{
    // Returns a random point in the camera defocus disk.
    auto p = random_in_unit_disk(rand_states);
    return _scene_info.camera_center + (p.x() * _defocus_disk_u) + (p.y() * _defocus_disk_v);
}