import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter


def plot_eigenvalue_spectrum(eigenvalues, eigenvectors, energies):
    plt.figure(figsize=(11,9))
    plt.axis([-1.1, 1.1, -1.1, 1.1])
    sc = plt.scatter(np.real(eigenvalues), np.imag(eigenvalues), c=energies, cmap='plasma', s=50)
    plt.colorbar(sc, label="Mode Energy")
    plt.xlabel("Re(λ)")
    plt.ylabel("Im(λ)")
    plt.title("Koopman Eigenvalue Spectrum")
    circle = plt.Circle((0,0), 1.0, color='k', fill=False, linestyle='--')
    plt.gca().add_artist(circle)
    plt.show()


def plot_vx_vz(vx, vz, norm_x, norm_z):
    plt.figure(figsize=(10,8))

    # --- vx ---
    plt.subplot(2,1,1)
    plt.imshow(vx.T, origin='lower', aspect='auto', cmap='viridis', norm=norm_x)
    plt.colorbar(label='vx')
    plt.title(f"vx")
    plt.xlabel("x-coordinate")
    plt.ylabel("z-coordinate")

    # --- vz ---
    plt.subplot(2,1,2)
    plt.imshow(vz.T, origin='lower', aspect='auto', cmap='plasma', norm=norm_z)
    plt.colorbar(label='vz')
    plt.title(f"vz")
    plt.xlabel("x-coordinate")
    plt.ylabel("z-coordinate")

    plt.tight_layout()
    plt.show()


def animate_trajectory(trajectory, file_name, norm_x, norm_z):
    n_timesteps = len(trajectory)
    fig, axes = plt.subplots(2, 1, figsize=(10, 8))

    # Initialize first frame
    vx = trajectory[0][0,:]
    vz = trajectory[0][1,:]

    im_vx = axes[0].imshow(vx.T, origin='lower', aspect='auto', cmap='viridis', norm=norm_x)
    axes[0].set_title("vx at time frame 0")
    axes[0].set_xlabel("x-coordinate")
    axes[0].set_ylabel("z-coordinate")
    fig.colorbar(im_vx, ax=axes[0], label="vx")

    im_vz = axes[1].imshow(vz.T, origin='lower', aspect='auto', cmap='plasma', norm=norm_z)
    axes[1].set_title("vz")
    axes[1].set_xlabel("x-coordinate")
    axes[1].set_ylabel("z-coordinate")
    fig.colorbar(im_vz, ax=axes[1], label="vz")

    plt.tight_layout()

    def update(frame):
        print(frame)
        vx = trajectory[frame][0,:]
        vz = trajectory[frame][1,:]

        im_vx.set_data(vx.T)
        im_vz.set_data(vz.T)

        axes[0].set_title(f"vx at time frame {frame}")
        #axes[1].set_title(f"vz")

        return [im_vx, im_vz]

    ani = FuncAnimation(fig, update,
                        frames=range(n_timesteps),
                        interval=30,
                        blit=False)

    # Save as GIF (requires Pillow)
    writer = PillowWriter(fps=20)
    ani.save(file_name+".gif", writer=writer)