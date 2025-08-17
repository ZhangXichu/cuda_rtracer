#include <obj_loader.cuh>

#include <cstring> 
#include <cuda_runtime.h> 

// helers
namespace {
    [[maybe_unused]]
    bool starts_with(const std::string& s, const char* prefix) {
        return s.rfind(prefix, 0) == 0;
    }

    // Parse an OBJ index token like "17", "17/3", "17/3/5", "17//5"
    // Returns the vertex index part as integer (may be negative in OBJ)
    int parse_face_index(const std::string& token) {
        size_t slash = token.find('/');
        const std::string head = (slash == std::string::npos) ? token : token.substr(0, slash);
        return std::stoi(head);
    }

    // Convert an OBJ index (1-based, negative allowed) to 0-based
    int obj_to_zero_based(int obj_index, int vertex_count) {
        if (obj_index > 0) return obj_index - 1;
        if (obj_index < 0) return vertex_count + obj_index;
        return 0;
    }
} // namespace

bool ObjLoader::load(const std::string& filename) {
    _vertices.clear();
    _faces.clear();

    std::ifstream in(filename);
    if (!in) return false;

    std::string line;
    int vcount = 0;

    while (std::getline(in, line)) {
        if (line.empty() || line[0] == '#') continue;

        std::istringstream ls(line);
        std::string tag;
        ls >> tag;
        if (tag == "v") { // load vertices
            float x, y, z;
            if (!(ls >> x >> y >> z)) continue; // malformed line -> skip
            _vertices.push_back(x);
            _vertices.push_back(y);
            _vertices.push_back(z);
            ++vcount;
        } else if (tag == "f") { // polygon face elements
            // collect all indices on the face
            std::vector<int> idx;
            std::string tok;
            while (ls >> tok) {
                int obj_idx = parse_face_index(tok);
                int zi = obj_to_zero_based(obj_idx, vcount);
                idx.push_back(zi);
            }
            if (idx.size() < 3) continue;

            // fan-triangulate: (0, i-1, i)
            for (size_t i = 2; i < idx.size(); ++i) {
                int a = idx[0], b = idx[i - 1], c = idx[i];
                // basic bounds guard
                if (a >= 0 && b >= 0 && c >= 0 &&
                    a < vcount && b < vcount && c < vcount) {
                    _faces.push_back(make_int3(a, b, c));
                }
            }
        }
        // ignore vt, vn, g, s, usemtl, etc.
    }

    return !_vertices.empty() && !_faces.empty();
}

void ObjLoader::normalize() {
    if (_vertices.empty()) return;

    float minx =  std::numeric_limits<float>::infinity();
    float miny =  std::numeric_limits<float>::infinity();
    float minz =  std::numeric_limits<float>::infinity();
    float maxx = -std::numeric_limits<float>::infinity();
    float maxy = -std::numeric_limits<float>::infinity();
    float maxz = -std::numeric_limits<float>::infinity();

    const size_t n = _vertices.size() / 3;
    for (size_t i = 0; i < n; ++i) {
        float x = _vertices[3*i+0];
        float y = _vertices[3*i+1];
        float z = _vertices[3*i+2];
        minx = std::min(minx, x); maxx = std::max(maxx, x);
        miny = std::min(miny, y); maxy = std::max(maxy, y);
        minz = std::min(minz, z); maxz = std::max(maxz, z);
    }

    const float cx = 0.5f * (minx + maxx);
    const float cy = 0.5f * (miny + maxy);
    const float cz = 0.5f * (minz + maxz);

    const float dx = maxx - minx;
    const float dy = maxy - miny;
    const float dz = maxz - minz;
    const float diag = std::sqrt(dx*dx + dy*dy + dz*dz);
    const float inv  = (diag > 0.f) ? (1.f / diag) : 1.f;

    for (size_t i = 0; i < n; ++i) {
        _vertices[3*i+0] = (_vertices[3*i+0] - cx) * inv;
        _vertices[3*i+1] = (_vertices[3*i+1] - cy) * inv;
        _vertices[3*i+2] = (_vertices[3*i+2] - cz) * inv;
    }
}

void ObjLoader::upload_to_device(float*& d_vertices, int& vcount,
                                 int3*& d_faces, int& fcount) const
{
    vcount = static_cast<int>(_vertices.size() / 3);
    fcount = static_cast<int>(_faces.size());

    // Managed memory keeps the call sites simple
    if (vcount > 0) {
        cudaMallocManaged(&d_vertices, _vertices.size() * sizeof(float));
        std::memcpy(d_vertices, _vertices.data(), _vertices.size() * sizeof(float));
    } else {
        d_vertices = nullptr;
    }

    if (fcount > 0) {
        cudaMallocManaged(&d_faces, _faces.size() * sizeof(int3));
        std::memcpy(d_faces, _faces.data(), _faces.size() * sizeof(int3));
    } else {
        d_faces = nullptr;
    }
}