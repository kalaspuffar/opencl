package org.ea.jocl;

import org.jocl.*;

import javax.imageio.ImageIO;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.*;

import static org.jocl.CL.*;
import static org.jocl.CL.clCreateCommandQueue;

public class BlurOpenCl {
    private static final int[] filterDelute = new int[] {
            1, 2, 1,
            2, 4, 2,
            1, 2, 1
    };

    private static String readFile(String fileName) {
        try {
            BufferedReader br = new BufferedReader(
                    new InputStreamReader(new FileInputStream(fileName)));
            StringBuffer sb = new StringBuffer();
            String line;
            while ((line = br.readLine()) != null) {
                sb.append(line).append("\n");
            }
            return sb.toString();
        } catch (IOException e) {
            e.printStackTrace();
            System.exit(1);
            return null;
        }
    }

    public static void main(String[] args) {
        try {
            String kernelFileName = "kernels/TestKernel.cu";
            String kernelSource = readFile(kernelFileName);

            CL.setExceptionsEnabled(true);

            // The platform, device type and device number
            // that will be used
            final int platformIndex = 0;
            final long deviceType = CL_DEVICE_TYPE_ALL;
            final int deviceIndex = 0;

            // Obtain the number of platforms
            int numPlatformsArray[] = new int[1];
            clGetPlatformIDs(0, null, numPlatformsArray);
            int numPlatforms = numPlatformsArray[0];

            // Obtain a platform ID
            cl_platform_id platforms[] = new cl_platform_id[numPlatforms];
            clGetPlatformIDs(platforms.length, platforms, null);
            cl_platform_id platform = platforms[platformIndex];

            // Obtain the number of devices for the platform
            int numDevicesArray[] = new int[1];
            clGetDeviceIDs(platform, deviceType, 0, null, numDevicesArray);
            int numDevices = numDevicesArray[0];

            // Obtain a device ID
            cl_device_id devices[] = new cl_device_id[numDevices];
            clGetDeviceIDs(platform, deviceType, numDevices, devices, null);
            cl_device_id device = devices[deviceIndex];

            // Initialize the context properties
            cl_context_properties contextProperties = new cl_context_properties();
            contextProperties.addProperty(CL_CONTEXT_PLATFORM, platform);

            // Create a context for the selected device
            cl_context context = clCreateContext(
                    contextProperties, 1, new cl_device_id[]{device},
                    null, null, null);

            // Create a command-queue for the selected device
            cl_command_queue commandQueue =
                    clCreateCommandQueue(context, device, 0, null);

            // Create the program from the source code
            cl_program program = clCreateProgramWithSource(context,
                    1, new String[]{ kernelSource }, null, null);

            // Build the program
            clBuildProgram(program, 0, null, null, null, null);

            // Create the kernel
            cl_kernel kernel = clCreateKernel(program, "sampleKernel", null);

            BufferedImage bi = ImageIO.read(new File("test.png"));
            int w = bi.getWidth();
            int h = bi.getHeight();
            int[] pixels = new int[w * h];
            int[] result = new int[w * h];

            BufferedImage grayscaleImg = new BufferedImage(w, h, BufferedImage.TYPE_BYTE_GRAY);
            Graphics g = grayscaleImg.getGraphics();
            g.drawImage(bi, 0, 0, null);
            g.dispose();

            pixels = grayscaleImg.getRaster().getPixels(0, 0, w, h, pixels);

            long start = System.currentTimeMillis();

            int numElements = w * h;

            Pointer srcA = Pointer.to(pixels);
            Pointer srcB = Pointer.to(filterDelute);
            Pointer dst = Pointer.to(result);

            cl_mem memObjects[] = new cl_mem[3];
            memObjects[0] = clCreateBuffer(context,
                    CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                    Sizeof.cl_int * numElements, srcA, null);
            memObjects[1] = clCreateBuffer(context,
                    CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                    Sizeof.cl_int * filterDelute.length, srcB, null);
            memObjects[2] = clCreateBuffer(context,
                    CL_MEM_READ_WRITE,
                    Sizeof.cl_int * numElements, null, null);

            // Set the arguments for the kernel
            clSetKernelArg(kernel, 0,
                    Sizeof.cl_int2, Pointer.to(new int[] {w, h}));
            clSetKernelArg(kernel, 1,
                    Sizeof.cl_mem, Pointer.to(memObjects[0]));
            clSetKernelArg(kernel, 2,
                    Sizeof.cl_mem, Pointer.to(memObjects[1]));
            clSetKernelArg(kernel, 3,
                    Sizeof.cl_mem, Pointer.to(memObjects[2]));

            // Set the work-item dimensions
            long global_work_size[] = new long[]{w, h};
            long local_work_size[] = new long[]{3, 3};

            // Execute the kernel
            clEnqueueNDRangeKernel(commandQueue, kernel, 2, null,
                    global_work_size, local_work_size, 0, null, null);

            // Read the output data
            clEnqueueReadBuffer(commandQueue, memObjects[2], CL_TRUE, 0,
                    Sizeof.cl_int * numElements, dst, 0, null, null);

            // Release kernel, program, and memory objects
            clReleaseMemObject(memObjects[0]);
            clReleaseMemObject(memObjects[1]);
            clReleaseMemObject(memObjects[2]);
            clReleaseKernel(kernel);
            clReleaseProgram(program);
            clReleaseCommandQueue(commandQueue);
            clReleaseContext(context);

            System.out.println("Time " + (System.currentTimeMillis() - start));

            grayscaleImg.getRaster().setPixels(0, 0, w, h, result);

            ImageIO.write(grayscaleImg, "PNG", new File("opencl-result.png"));
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}