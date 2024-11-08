	.text
	.file	"boas_module"
	.section	.rodata.cst8,"aM",@progbits,8
	.p2align	3, 0x0                          # -- Begin function main
.LCPI0_0:
	.quad	0x4024000000000000              # double 10
.LCPI0_1:
	.quad	0x4034800000000000              # double 20.5
.LCPI0_2:
	.quad	0x4049800000000000              # double 51
	.text
	.globl	main
	.p2align	4
	.type	main,@function
main:                                   # @main
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rax
	.cfi_def_cfa_offset 16
	leaq	.L__unnamed_1(%rip), %rdi
	movsd	.LCPI0_0(%rip), %xmm0           # xmm0 = [1.0E+1,0.0E+0]
	movb	$1, %al
	callq	printf@PLT
	leaq	.L__unnamed_2(%rip), %rdi
	movsd	.LCPI0_1(%rip), %xmm0           # xmm0 = [2.05E+1,0.0E+0]
	movb	$1, %al
	callq	printf@PLT
	leaq	.L__unnamed_3(%rip), %rdi
	xorl	%eax, %eax
	callq	printf@PLT
	leaq	.L__unnamed_4(%rip), %rdi
	xorl	%eax, %eax
	callq	printf@PLT
	leaq	.L__unnamed_5(%rip), %rdi
	movsd	.LCPI0_2(%rip), %xmm0           # xmm0 = [5.1E+1,0.0E+0]
	movb	$1, %al
	callq	printf@PLT
	xorl	%eax, %eax
	popq	%rcx
	.cfi_def_cfa_offset 8
	retq
.Lfunc_end0:
	.size	main, .Lfunc_end0-main
	.cfi_endproc
                                        # -- End function
	.type	.L__unnamed_1,@object           # @0
	.section	.rodata.str1.1,"aMS",@progbits,1
.L__unnamed_1:
	.asciz	"%.6g\n"
	.size	.L__unnamed_1, 6

	.type	.L__unnamed_2,@object           # @1
.L__unnamed_2:
	.asciz	"%.6g\n"
	.size	.L__unnamed_2, 6

	.type	.L__unnamed_3,@object           # @2
.L__unnamed_3:
	.asciz	"Boas\n"
	.size	.L__unnamed_3, 6

	.type	.L__unnamed_4,@object           # @3
.L__unnamed_4:
	.asciz	"true\n"
	.size	.L__unnamed_4, 6

	.type	.L__unnamed_5,@object           # @4
.L__unnamed_5:
	.asciz	"%.6g\n"
	.size	.L__unnamed_5, 6

	.section	".note.GNU-stack","",@progbits
