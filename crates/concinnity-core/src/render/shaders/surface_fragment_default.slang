// The engine's own surface hook, spliced at SURFACE_FRAGMENT unless a world
// Shader supplies a `fragment` file. A world file defines the same function
// and may call `shade_surface` itself to start from the engine's lighting.
float4 shade(VertexOut in, GpuObjectData od)
{
    return shade_surface(in, od);
}
