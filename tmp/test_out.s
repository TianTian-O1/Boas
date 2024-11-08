	.text
	.file	"boas_module"
	.section	.rodata.cst8,"aM",@progbits,8
	.p2align	3, 0x0                          # -- Begin function main
.LCPI0_0:
	.quad	0x4045000000000000              # double 42
.LCPI0_1:
	.quad	0x40091eb851eb851f              # double 3.1400000000000001
	.text
	.globl	main
	.p2align	4
	.type	main,@function
main:                                   # @main
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rax
	.cfi_def_cfa_offset 16
	movsd	.LCPI0_0(%rip), %xmm0           # xmm0 = [4.2E+1,0.0E+0]
	movl	$.Lfmt, %edi
	movb	$1, %al
	callq	printf@PLT
	movsd	.LCPI0_1(%rip), %xmm0           # xmm0 = [3.1400000000000001E+0,0.0E+0]
	movl	$.Lfmt.1, %edi
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
	.type	.Lfmt,@object                   # @fmt
	.section	.rodata.str1.1,"aMS",@progbits,1
.Lfmt:
	.asciz	"%.2f\n"
	.size	.Lfmt, 6

	.type	.Lfmt.1,@object                 # @fmt.1
.Lfmt.1:
	.asciz	"%.2f\n"
	.size	.Lfmt.1, 6

	.section	".note.GNU-stack","",@progbits
