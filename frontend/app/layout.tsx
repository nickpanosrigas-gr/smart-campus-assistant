import type { Metadata } from "next";
import { Outfit } from "next/font/google";
import "./globals.css";

const outfit = Outfit({ subsets: ["latin"], variable: "--font-outfit" });

export const metadata: Metadata = {
  title: "Smart Campus Assistant",
  description: "Intelligent Assistant for Smart Campus Management",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" className="dark">
      {/* Background: #0A0A0A (bottom) to #0A664F (top)
        Text: #A3B8B2
      */}
      <body
        className={`${outfit.variable} font-sans antialiased bg-gradient-to-b from-[#0A664F] to-[#0A0A0A] text-[#A3B8B2] min-h-screen overflow-hidden`}
      >
        {children}
      </body>
    </html>
  );
}