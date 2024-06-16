#include <camera.cuh>

__device__ Hittable** sphere_lst;
__device__ Hittable* world;
__device__ Metal* metal;
__device__ Lambertian* lambertian;
__device__ Dielectric* dielectric;

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
            // current_ray = Ray(record.p, direction);
            Ray scattered;
            Color attenuation;
            if (record.material->scatter(current_ray, record, attenuation, scattered, rand_states))
            {
                accumulated_color = accumulated_color * attenuation;
                current_ray = scattered;
            }
            // accumulated_color *= 0.5;
            
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
    // Construct a camera ray originating from the origin and directed at randomly sampled
        // point around the pixel location i, j.
    auto offset = sample_square(rand_states);
    auto pixel_sample = _scene_info.pixel00_loc
                        + ((i + offset.x()) * _scene_info.pixel_delta_u)
                        + ((j + offset.y()) * _scene_info.pixel_delta_v);

    auto ray_origin = _scene_info.camera_center;
    auto ray_direction = pixel_sample - ray_origin;

    return Ray(ray_origin, ray_direction);
}

int Camera::get_img_height() const
{
    return _img_height;
}

SceneInfo Camera::get_scene_info() const
{
    return _scene_info;
}