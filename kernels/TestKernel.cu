__kernel void sampleKernel(
    const int2 imageSize,
    __global const int *img,
    __global const int *filter,
    __global int *result)
{
    int FILTER_SIZE = 3;
    int x = get_global_id(0);
    int y = get_global_id(1);

    int width = imageSize.x;
    int height = imageSize.y;

    int sum = 0;
    for(int filterY=0; filterY < FILTER_SIZE; filterY++) {
        for(int filterX=0; filterX < FILTER_SIZE; filterX++) {
            sum += img[ mul24((y - 1 + filterY), width) + x - 1 + filterX ] * filter[ (filterY * FILTER_SIZE) + filterX ];
        }
    }

    if(y + 1 < height && x + 1 < width) {
        result[mul24((y), width) + x] = sum / 16;
    }
}