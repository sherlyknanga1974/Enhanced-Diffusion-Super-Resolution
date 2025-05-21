This project proposes an Enhanced Diffusion-Based Super-Resolution Model (EDSRM) designed to generate high-fidelity, high-resolution images from low-resolution inputs. It integrates advanced diffusion model techniques such as:
•	Differentiable Augmentation (DiffAugment) for stable training,
•	Probability Flow Sampling (PFS) for efficient inference,
•	Hybrid Parameterization for adaptive noise estimation,
•	De-noising Diffusion Implicit Models (DDIM) for faster convergence.
OVERALL BLOCKDIAGRAM 

![image](https://github.com/user-attachments/assets/a12dd02a-cf61-4fab-9258-b5b256fde29f)


The system operates in a multi-stage framework to enhance image resolution while preserving texture and reducing computational cost, suitable for applications like medical imaging, satellite image enhancement, and real-time image generation.
EDSRM addresses key limitations in traditional and diffusion-based super-resolution methods, such as:
•	Over-smoothing and loss of fine texture,
•	Slow inference due to iterative sampling,
•	Instability in training and poor perceptual quality.
The project provides:
•	Real-time inference (8.54 ms),
•	High visual fidelity (PSNR of 39.94, SSIM of 0.854),
•	Low perceptual error (LPIPS of 0.082, FID of 2.03),
•	Suitability for real-world tasks such as medical imaging and remote sensing.
 This project includes Python, PyTorch, and dependencies like torchvision and numpy. Development and experimentation were conducted using PyTorch on NVIDIA A100 GPUs,
 The project is implemented with datasets such as ImageNet 256×256 and LSUN Bedroom
The project is done with the help of Helen Sulochana helen@sxcce.edu.in.
The project is authored and maintained by: Sherly Kanaga Priya sherlyknanga1974@gmail.com

