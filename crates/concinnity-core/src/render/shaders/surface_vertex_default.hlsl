// The engine's own vertex hook, spliced at SURFACE_VERTEX unless a world
// Shader supplies a `vertex` file of its own. A world file defines the same
// function; this one is what every draw without a Shader runs.
VertexOut transform(float4x4 model, float3 pos, float3 normal, float3 tangent,
                    float3 color, float2 uv)
{
    return project_vertex(model, pos, normal, tangent, color, uv);
}
